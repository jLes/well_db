"""
FastAPI application for serving well_db data.
"""
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
import geopandas as gpd
from shapely.geometry import Point, Polygon

from well_db.database import get_db, init_db, DB_PATH, get_all_apis, db_session, upsert_well
from well_db.models import (
    WellData,
    WellDataSchema,
    PolygonSearchRequest,
    PolygonSearchResponse,
    DatabaseStatusResponse,
    ScrapeStatusResponse,
    ScrapeStartResponse,
    DEFAULT_CSV_PATH,
)
from well_db.scraper import (
    is_valid_api_format,
    get_random_api,
    load_api_numbers_from_csv,
    scrape_batch,
)


# =============================================================================
# Scrape Job State (in-memory for simplicity; could use Redis for production)
# =============================================================================

class ScrapeJobState:
    """Tracks the state of a background scrape job."""

    def __init__(self):
        self.job_id: Optional[str] = None
        self.status: str = "idle"  # idle, running, completed, failed, stopped
        self.total: int = 0
        self.completed: int = 0
        self.failed: int = 0
        self.current_api: Optional[str] = None
        self.errors: list[str] = []
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.stop_requested: bool = False

    def start(self, job_id: str, total: int):
        self.job_id = job_id
        self.status = "running"
        self.total = total
        self.completed = 0
        self.failed = 0
        self.current_api = None
        self.errors = []
        self.started_at = datetime.now(timezone.utc)
        self.finished_at = None
        self.stop_requested = False

    def update(self, current_api: str):
        self.current_api = current_api

    def record_success(self):
        self.completed += 1

    def record_failure(self, api: str, error: str):
        self.failed += 1
        self.errors.append(f"{api}: {error}")

    def request_stop(self):
        self.stop_requested = True

    def should_stop(self) -> bool:
        return self.stop_requested

    def finish(self, status: str = "completed"):
        self.status = status
        self.current_api = None
        self.finished_at = datetime.now(timezone.utc)
        self.stop_requested = False

    @property
    def progress_percent(self) -> float:
        if self.total == 0:
            return 0.0
        return round((self.completed + self.failed) / self.total * 100, 1)

    def to_response(self) -> ScrapeStatusResponse:
        return ScrapeStatusResponse(
            job_id=self.job_id or "",
            status=self.status,
            total=self.total,
            completed=self.completed,
            failed=self.failed,
            progress_percent=self.progress_percent,
            current_api=self.current_api,
            errors=self.errors[-10:],  # Last 10 errors only
            started_at=self.started_at.isoformat() if self.started_at else None,
            finished_at=self.finished_at.isoformat() if self.finished_at else None,
        )


# Global scrape job state
scrape_state = ScrapeJobState()

# Tag metadata for organizing endpoints in docs
tags_metadata = [
    {
        "name": "Required",
        "description": "Required endpoints: well lookup and polygon search.",
    },
    {
        "name": "Database & Scraping",
        "description": "Endpoints for database status, scraping operations, and data management.",
    },
    {
        "name": "Utilities",
        "description": "Health checks and other utility endpoints.",
    },
]

# Initialize FastAPI app
app = FastAPI(
    title="WellDB API",
    description="API for querying New Mexico oil and gas well data",
    version="0.0.1",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "tryItOutEnabled": True,  # All endpoints ready to test immediately
        "persistAuthorization": True,  # Remember auth between page refreshes
        "filter": True,  # Enable filtering/search in the docs
    },
)


@app.on_event("startup")
def startup():
    """Initialize database on startup."""
    init_db()


@app.get("/", tags=["Utilities"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "WellDB API"}


### Assignment Required Endpoints


@app.get("/well", response_model=WellDataSchema, tags=["Required"])
def get_well(
    api_number: str = Query(..., description="The well's API identifier (e.g., '30-015-25325')"),
    db: Session = Depends(get_db)
):
    """
    Get all data for a single well by API number.

    Queries the database for the specified API number and returns all available
    data if found. Returns 404 if the well is not in the database.

    Returns:
        All available data for the specified well
    """
    # Validate API format
    if not is_valid_api_format(api_number):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid API format: '{api_number}'. Expected format: XX-XXX-XXXXX"
        )

    # Query database
    well = db.query(WellData).filter(WellData.api == api_number).first()

    if not well:
        raise HTTPException(
            status_code=404,
            detail=f"Well {api_number} not found in database"
        )

    return well


@app.get("/polygon-search", response_model=PolygonSearchResponse, tags=["Required"])
def search_polygon(
    polygon: str = Query(
        ...,
        description="Polygon vertices as '[(lat1,lon1),(lat2,lon2),...]' (minimum 3 points)",
        example="[(32.81,-104.19),(32.66,-104.32),(32.54,-104.24),(32.50,-104.03)]"
    ),
    db: Session = Depends(get_db)
):
    """
    Search for wells within a polygon.

    The polygon is specified as a list of (latitude, longitude) tuples in the format:
    [(lat1,lon1),(lat2,lon2),(lat3,lon3),...]

    Args:
        polygon: Polygon vertices as "[(lat1,lon1),(lat2,lon2),...]"

    Returns:
        List of API numbers for wells within the polygon
    """
    # Parse polygon string in format: [(lat1,lon1),(lat2,lon2),...]
    try:
        # Extract all (lat,lon) pairs from the string
        pattern = r'\(([^)]+)\)'
        matches = re.findall(pattern, polygon)

        if not matches:
            raise ValueError("No coordinate pairs found")

        points = []
        for match in matches:
            parts = match.split(',')
            if len(parts) != 2:
                raise ValueError(f"Invalid coordinate pair: ({match})")
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            points.append((lat, lon))

        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 points")

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid polygon format. Use '[(lat1,lon1),(lat2,lon2),...]'. Error: {e}"
        )

    return _search_polygon(points, db)


# Extra Well Endpoints (not required by assignment)

async def _scrape_single_well(api_number: str) -> Optional[dict]:
    """Helper to scrape a single well using the WellScraper."""
    from well_db.scraper import WellScraper

    async with WellScraper(headless=True) as scraper:
        return await scraper.scrape_well(api_number)


@app.get("/wells/random", response_model=WellDataSchema, tags=["Database & Scraping"])
async def get_random_well(
    scrape_if_missing: bool = Query(
        True,
        description="If well not in DB, scrape it from the website"
    ),
    save_to_db: bool = Query(
        False,
        description="Save scraped data to database (only applies if scraping occurs)"
    ),
    db: Session = Depends(get_db)
):
    """
    Get a random well from the CSV list.

    Picks a random API number from the CSV file and returns its data.
    If the well is not in the database, it will be scraped from the website
    (unless scrape_if_missing=false).

    Returns:
        Data for a randomly selected well
    """
    api_number = get_random_api()

    if not api_number:
        raise HTTPException(
            status_code=500,
            detail="Could not get random API - CSV file may be missing"
        )

    # Check database first
    well = db.query(WellData).filter(WellData.api == api_number).first()

    if well:
        return well

    # Well not in DB
    if not scrape_if_missing:
        raise HTTPException(
            status_code=404,
            detail=f"Random well {api_number} not found in database. Use ?scrape_if_missing=true to fetch from website."
        )

    # Scrape from website
    well_data = await _scrape_single_well(api_number)

    if not well_data:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to scrape well {api_number} from website"
        )

    # Optionally save to database
    if save_to_db and DB_PATH.exists():
        with db_session() as db_write:
            upsert_well(db_write, well_data)

    return WellDataSchema(**well_data)


@app.get("/well/scrape", response_model=WellDataSchema, tags=["Database & Scraping"])
async def scrape_well(
    api_number: str = Query(..., description="The well's API identifier (e.g., '30-015-25325')"),
    save_to_db: bool = Query(
        False,
        description="Save the scraped data to the database"
    ),
):
    """
    Force scrape a well from the website (bypasses database).

    This endpoint always fetches fresh data from the NM OCD website,
    regardless of whether the well exists in the database.

    Args:
        api_number: The well's API identifier (e.g., "30-015-25325")
        save_to_db: If True, also save/update the scraped data in the database

    Returns:
        Freshly scraped data for the specified well
    """
    # Validate API format
    if not is_valid_api_format(api_number):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid API format: '{api_number}'. Expected format: XX-XXX-XXXXX"
        )

    # Scrape from website
    well_data = await _scrape_single_well(api_number)

    if not well_data:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to scrape well {api_number} from website"
        )

    # Optionally save to database
    if save_to_db:
        with db_session() as db:
            upsert_well(db, well_data)

    return WellDataSchema(**well_data)


@app.get("/wells", response_model=list[WellDataSchema], tags=["Database & Scraping"])
def list_wells(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    db: Session = Depends(get_db)
):
    """
    List all wells with pagination.

    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return

    Returns:
        List of wells
    """
    wells = db.query(WellData).offset(skip).limit(limit).all()
    return wells


@app.get("/wells/count", tags=["Database & Scraping"])
def count_wells(db: Session = Depends(get_db)):
    """Get total count of wells in database."""
    count = db.query(WellData).count()
    return {"count": count}

# A more RESTful POST-based polygon search endpoint . . . 
@app.post("/polygon-search", response_model=PolygonSearchResponse, tags=["Database & Scraping"])
def search_polygon_post(request: PolygonSearchRequest, db: Session = Depends(get_db)):
    """
    Search for wells within a polygon (POST method).

    Args:
        request: Polygon search request with list of (lat, lon) vertices

    Returns:
        List of API numbers for wells within the polygon
    """
    return _search_polygon(request.polygon, db)


def _search_polygon(polygon_points: list[tuple[float, float]], db: Session) -> PolygonSearchResponse:
    """
    Core polygon search logic using GeoPandas for proper geodetic handling.

    Uses bounding box pre-filtering for performance, then GeoPandas with
    WGS84 CRS (EPSG:4326) for accurate point-in-polygon testing.

    Args:
        polygon_points: List of (latitude, longitude) tuples defining the polygon
        db: Database session

    Returns:
        PolygonSearchResponse with matching API numbers
    """
    # Extract lat/lon bounds for bounding box pre-filter
    lats = [p[0] for p in polygon_points]
    lons = [p[1] for p in polygon_points]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Pre-filter candidates using bounding box (fast SQL query)
    candidates = db.query(WellData).filter(
        WellData.latitude.isnot(None),
        WellData.longitude.isnot(None),
        WellData.latitude.between(min_lat, max_lat),
        WellData.longitude.between(min_lon, max_lon)
    ).all()

    if not candidates:
        return PolygonSearchResponse(api_numbers=[], count=0)

    # Create polygon in proper GIS format: (lon, lat) for x, y
    # Input is (lat, lon), so we need to swap for Shapely/GeoPandas
    polygon_coords_xy = [(lon, lat) for lat, lon in polygon_points]
    search_polygon = Polygon(polygon_coords_xy)

    # Create GeoDataFrame for the polygon with WGS84 CRS
    polygon_gdf = gpd.GeoDataFrame(
        geometry=[search_polygon],
        crs="EPSG:4326"  # WGS84 - standard lat/lon CRS
    )

    # Create GeoDataFrame for well points
    well_points = [Point(well.longitude, well.latitude) for well in candidates]
    well_apis = [well.api for well in candidates]

    wells_gdf = gpd.GeoDataFrame(
        {"api": well_apis},
        geometry=well_points,
        crs="EPSG:4326"
    )

    # Spatial join to find points within polygon
    # This uses proper geodetic calculations via GeoPandas
    matches = gpd.sjoin(wells_gdf, polygon_gdf, predicate="within")
    matching_apis = matches["api"].tolist()

    return PolygonSearchResponse(
        api_numbers=matching_apis,
        count=len(matching_apis)
    )


### Database Status & Scrape Endpoints


@app.get("/db/status", response_model=DatabaseStatusResponse, tags=["Database & Scraping"])
def get_db_status(
    csv_path: Optional[str] = Query(None, description="Path to CSV file (uses default if not provided)"),
    db: Session = Depends(get_db)
):
    """
    Get database status and comparison with CSV.

    Returns info about:
    - Whether database exists and row count
    - CSV entry count
    - Missing entries (in CSV but not in DB)
    - Column names
    - Whether all API numbers are unique
    """
    # Check if DB file exists
    db_exists = DB_PATH.exists()

    # Get row count
    row_count = db.query(WellData).count() if db_exists else 0

    # Get all APIs in DB
    db_apis = set(get_all_apis(db)) if db_exists else set()

    # Load CSV
    csv_file = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_file}")

    csv_apis = load_api_numbers_from_csv(csv_file)
    csv_count = len(csv_apis)

    # Check for missing APIs
    missing_apis = set(csv_apis) - db_apis
    missing_count = len(missing_apis)

    # Check uniqueness (in DB)
    unique_apis = row_count == len(db_apis)

    # Get column names from model
    columns = [c.name for c in WellData.__table__.columns]

    return DatabaseStatusResponse(
        db_exists=db_exists,
        db_path=str(DB_PATH),
        row_count=row_count,
        csv_count=csv_count,
        missing_count=missing_count,
        is_complete=(missing_count == 0 and row_count > 0),
        columns=columns,
        unique_apis=unique_apis,
    )


@app.get("/scrape/status", response_model=ScrapeStatusResponse, tags=["Database & Scraping"])
def get_scrape_status():
    """Get the current status of the scrape job."""
    return scrape_state.to_response()


async def _run_scrape(apis_to_scrape: list[str], job_id: str, concurrency: int = 3):
    """Background task to scrape missing APIs using the optimized scrape_batch."""
    scrape_state.start(job_id, len(apis_to_scrape))

    def on_progress(api: str, _current: int, _total: int) -> None:
        """Called when a scrape attempt starts."""
        scrape_state.update(api)
        # Check for stop request (will be handled after current batch completes)
        if scrape_state.should_stop():
            raise StopIteration("Stop requested")

    def on_result(well_data: dict) -> None:
        """Called on successful scrape - save to database."""
        with db_session() as db:
            upsert_well(db, well_data)
        scrape_state.record_success()

    def on_error(api: str, error_msg: str) -> None:
        """Called on failed scrape after all retries exhausted."""
        scrape_state.record_failure(api, error_msg)

    try:
        await scrape_batch(
            api_numbers=apis_to_scrape,
            concurrency=concurrency,
            on_progress=on_progress,
            on_result=on_result,
            on_error=on_error,
        )

        if scrape_state.should_stop():
            scrape_state.finish("stopped")
        else:
            scrape_state.finish("completed")

    except StopIteration:
        scrape_state.finish("stopped")
    except Exception as e:
        scrape_state.record_failure("FATAL", str(e))
        scrape_state.finish("failed")


@app.post("/scrape/start", response_model=ScrapeStartResponse, tags=["Database & Scraping"])
async def start_scrape(
    background_tasks: BackgroundTasks,
    csv_path: Optional[str] = Query(None, description="Path to CSV file"),
    concurrency: int = Query(3, ge=1, le=10, description="Number of concurrent scrape workers"),
    db: Session = Depends(get_db)
):
    """
    Start a background scrape job for missing APIs.

    Only scrapes APIs that are in the CSV but not in the database.
    Uses the worker pool pattern with proper rate limiting (1.5s between requests per worker).
    Returns immediately with job ID; use /scrape/status to monitor progress.
    """
    # Check if scrape is already running
    if scrape_state.status == "running":
        raise HTTPException(
            status_code=409,
            detail=f"Scrape job already running (job_id: {scrape_state.job_id})"
        )

    # Load CSV
    csv_file = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_file}")

    csv_apis = load_api_numbers_from_csv(csv_file)

    # Get existing APIs in DB
    db_apis = set(get_all_apis(db))

    # Find missing APIs
    apis_to_scrape = [api for api in csv_apis if api not in db_apis]

    if not apis_to_scrape:
        return ScrapeStartResponse(
            job_id="",
            message="Database is already complete. No APIs to scrape.",
            apis_to_scrape=0
        )

    # Generate job ID and start background task
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_run_scrape, apis_to_scrape, job_id, concurrency)

    return ScrapeStartResponse(
        job_id=job_id,
        message=f"Scrape job started with {concurrency} workers. Use /scrape/status to monitor progress.",
        apis_to_scrape=len(apis_to_scrape)
    )


@app.post("/scrape/stop", tags=["Database & Scraping"])
def stop_scrape():
    """
    Request to stop the current scrape job.

    Note: This sets a flag; the job will stop after the current API completes.
    """
    if scrape_state.status != "running":
        raise HTTPException(status_code=400, detail="No scrape job is currently running")

    scrape_state.request_stop()
    return {"message": "Scrape job stop requested", "job_id": scrape_state.job_id}


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
