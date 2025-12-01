"""
Database connection, session management, and CRUD operations for well_db

This module provides:
- Database engine and session configuration
- CRUD operations (Create, Read, Update, Delete)
- Query helpers with filtering, sorting, and pagination
- Bulk operations for efficient data loading
- Context manager for safe session handling
"""
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator

from sqlalchemy import create_engine, desc, asc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from well_db.models import Base, WellData

# Configure logging
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "api_well_data.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite with FastAPI
    echo=False,  # True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


### Database Initialization & Management

def init_db() -> None:
    """Create all database tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized at {DB_PATH}")


def drop_db() -> None:
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


def reset_db() -> None:
    """Drop and recreate all tables."""
    drop_db()
    init_db()
    logger.warning("Database reset complete")


### Session Management
## Not used yet, but in anticipation of FastAPI integration
def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get a database session.
    Yields a session and ensures it's closed after use.

    Usage in FastAPI:
        @app.get("/wells")
        def get_wells(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """
    Get a database session for non-FastAPI use (e.g., scraping, CLI tools).

    IMPORTANT: Caller is responsible for closing the session!
    Prefer using db_session() context manager instead.
    """
    return SessionLocal()

@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Automatically handles commit/rollback and cleanup.

    Usage:
        with db_session() as db:
            well = db.query(WellData).first()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()

### CRUD Ops

def create_well(db: Session, well_data: dict) -> WellData:
    """
    Create a new well record.

    Args:
        db: Database session
        well_data: Dictionary of well attributes

    Returns:
        Created WellData instance

    Raises:
        IntegrityError: If well with same API already exists
    """
    well = WellData(**well_data)
    db.add(well)
    db.commit()
    db.refresh(well)
    logger.debug(f"Created well: {well.api}")
    return well


def get_well(db: Session, api: str) -> Optional[WellData]:
    """
    Get a well by API number.

    Args:
        db: Database session
        api: Well API number

    Returns:
        WellData instance or None if not found
    """
    return db.query(WellData).filter(WellData.api == api).first()


def get_wells(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "api",
    sort_order: str = "asc"
) -> list[WellData]:
    """
    Get multiple wells with pagination and sorting.

    Args:
        db: Database session
        skip: Number of records to skip (offset)
        limit: Maximum number of records to return
        sort_by: Field to sort by (default: "api")
        sort_order: "asc" or "desc" (default: "asc")

    Returns:
        List of WellData instances
    """
    query = db.query(WellData)

    # Apply sorting
    if hasattr(WellData, sort_by):
        column = getattr(WellData, sort_by)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(column))
        else:
            query = query.order_by(asc(column))

    return query.offset(skip).limit(limit).all()


def update_well(db: Session, api: str, updates: dict) -> Optional[WellData]:
    """
    Update an existing well record.

    Args:
        db: Database session
        api: Well API number to update
        updates: Dictionary of fields to update

    Returns:
        Updated WellData instance or None if not found
    """
    well = get_well(db, api)
    if well is None:
        return None

    for key, value in updates.items():
        if hasattr(well, key) and key != "api":  # Don't allow API ID change
            setattr(well, key, value)

    db.commit()
    db.refresh(well)
    logger.debug(f"Updated well: {api}")
    return well


def upsert_well(db: Session, well_data: dict) -> tuple[WellData, bool]:
    """
    Insert or update a well record.

    Args:
        db: Database session
        well_data: Dictionary of well attributes (must include 'api')

    Returns:
        Tuple of (WellData instance, created: bool)
        created is True if new record, False if updated
    """
    api = well_data.get("api")
    if not api:
        raise ValueError("well_data must include 'api' field")

    existing = get_well(db, api)
    if existing:
        updated = update_well(db, api, well_data)
        return updated, False
    else:
        created = create_well(db, well_data)
        return created, True


def delete_well(db: Session, api: str) -> bool:
    """
    Delete a well record.

    Args:
        db: Database session
        api: Well API number to delete

    Returns:
        True if deleted, False if not found
    """
    well = get_well(db, api)
    if well is None:
        return False

    db.delete(well)
    db.commit()
    logger.debug(f"Deleted well: {api}")
    return True


### Query Helpers

def search_wells(
    db: Session,
    operator: Optional[str] = None,
    status: Optional[str] = None,
    well_type: Optional[str] = None,
    mineral_owner: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "api",
    sort_order: str = "asc"
) -> list[WellData]:
    """
    Search wells with multiple filter criteria.

    Args:
        db: Database session
        status: Filter by status (exact match)
        well_type: Filter by well type (exact match)
        mineral_owner: Filter by mineral owner (exact match)
        skip: Pagination offset
        limit: Pagination limit
        sort_by: Field to sort by
        sort_order: "asc" or "desc"

    Returns:
        List of matching WellData instances
    """
    query = db.query(WellData)

    # Apply filters
    if status:
        query = query.filter(WellData.status == status)
    if well_type:
        query = query.filter(WellData.well_type == well_type)
    if mineral_owner:
        query = query.filter(WellData.mineral_owner == mineral_owner)

    # Apply sorting
    if hasattr(WellData, sort_by):
        column = getattr(WellData, sort_by)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(column))
        else:
            query = query.order_by(asc(column))

    return query.offset(skip).limit(limit).all()


def get_wells_by_status(db: Session, status: str) -> list[WellData]:
    """Get all wells with a specific status."""
    return db.query(WellData).filter(WellData.status == status).all()

# In prparation for polygon boundary queries
# Will probably end up using geopandas, instead of direct lat/lon value queries
def get_wells_in_bbox(
    db: Session,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> list[WellData]:
    """
    Get wells within a bounding box.

    Args:
        db: Database session
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        min_lon: Minimum longitude
        max_lon: Maximum longitude

    Returns:
        List of WellData instances within the bounding box
    """
    return db.query(WellData).filter(
        WellData.latitude >= min_lat,
        WellData.latitude <= max_lat,
        WellData.longitude >= min_lon,
        WellData.longitude <= max_lon
    ).all()


### Counting 

def count_wells(db: Session) -> int:
    """Get total count of wells in database."""
    return db.query(func.count(WellData.api)).scalar()


def count_wells_by_status(db: Session) -> dict[str, int]:
    """Get count of wells grouped by status."""
    results = db.query(
        WellData.status,
        func.count(WellData.api)
    ).group_by(WellData.status).all()

    return {status or "Unknown": count for status, count in results}


def count_wells_by_operator(db: Session, limit: int = 10) -> dict[str, int]:
    """Get count of wells grouped by operator (top N)."""
    results = db.query(
        WellData.operator,
        func.count(WellData.api).label('count')
    ).group_by(WellData.operator).order_by(
        desc('count')
    ).limit(limit).all()

    return {operator or "Unknown": count for operator, count in results}


def get_unique_statuses(db: Session) -> list[str]:
    """Get list of unique well statuses."""
    results = db.query(WellData.status).distinct().all()
    return [r[0] for r in results if r[0]]


def get_unique_operators(db: Session) -> list[str]:
    """Get list of unique operators."""
    results = db.query(WellData.operator).distinct().all()
    return [r[0] for r in results if r[0]]

### Utilities

def well_exists(db: Session, api: str) -> bool:
    """Check if a well exists in the database."""
    return db.query(WellData.api).filter(WellData.api == api).first() is not None


def get_all_apis(db: Session) -> list[str]:
    """Get list of all API numbers in database."""
    results = db.query(WellData.api).all()
    return [r[0] for r in results]


def export_wells_to_dicts(db: Session, limit: Optional[int] = None) -> list[dict]:
    """
    Export all wells as list of dictionaries.

    Args:
        db: Database session
        limit: Optional limit on number of records

    Returns:
        List of well dictionaries
    """
    query = db.query(WellData)
    if limit:
        query = query.limit(limit)

    return [well.to_dict() for well in query.all()]
