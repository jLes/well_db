# WellDB

A web scraper and REST API for New Mexico Oil Conservation Division (OCD) well data.

### DATABASE FILE: api_well_data.db
### Polygon Query Results: polygon_query_results.csv 

## Setup

### Option 1: uv (Recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and set up environment
uv sync
uv run playwright install chromium
```

### Option 2: Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
playwright install chromium
```

### Option 3: Docker

```bash
docker compose up --build

OR , if venv or UV venv activated:

[UV run] python -m well_db docker up --build
```

This starts the API server at `http://localhost:8000`. The Playwright browser is included in the image.

## Getting Started

### Scrape Well Data

Populate the database from the CSV file containing API numbers:

```bash
# Using uv
uv run python -m well_db scrape

# With options
uv run python -m well_db scrape --concurrency 3 --missing
```

The `--missing` flag only scrapes APIs not already in the database. Use `--force` to re-scrape everything.

### Start the API Server

```bash
uv run python -m well_db serve
```

Server runs at `http://127.0.0.1:8000`. API docs available at `/docs`.

### Query Wells in a Polygon (CLI)

```bash
# Use the test polygon from the assignment
uv run python -m well_db polygon test -o results.csv

# Custom polygon
uv run python -m well_db polygon "[(32.81,-104.19),(32.66,-104.32),(32.54,-104.24)]" -o results.csv
```

### Other Commands

```bash
uv run python -m well_db delete --yes    # Delete the database
uv run python -m well_db docker up       # Start Docker containers
uv run python -m well_db docker down     # Stop Docker containers
```

## API Endpoints

### Required Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/well?api_number=XX-XXX-XXXXX` | Get all data for a single well |
| GET | `/polygon-search?polygon=[(lat,lon),...]` | Find wells within a polygon |

### Database and Scraping

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/db/status` | Database status and CSV comparison |
| POST | `/scrape/start` | Start background scrape job |
| GET | `/scrape/status` | Monitor scrape progress |
| POST | `/scrape/stop` | Stop running scrape job |
| GET | `/wells` | List all wells (paginated) |
| GET | `/wells/count` | Total well count |

### Utilities

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/well/scrape?api_number=...` | Force scrape a single well |
| GET | `/wells/random` | Get a random well (scrapes if missing) |
| POST | `/polygon-search` | Polygon search with JSON body |

## Implementation Notes

### Scraper Architecture

The scraper uses Playwright with headless Chromium. 

Key scraper features:
- **Element-based waiting**: Uses `wait_for_selector("#general_information")` [General Well Information] instead of fixed delays for reliable page load detection
- **Targeted extraction**: Extracts only the `fieldset.data_container` element rather than full page text
- **Worker pool concurrency**: Uses `asyncio.Queue` with configurable workers instead of semaphore-based approaches

### Concurrency Strategy

I attempted to use `asyncio.gather()` with a semaphore, which caused request clustering. The final implementation uses a worker pool pattern where N workers pull from a shared queue, each maintaining a 1.5-second delay between their own requests. This provides consistent rate limiting without overwhelming the target server.

### Polygon Search

Uses GeoPandas with WGS84 CRS (EPSG:4326) for geodetically-correct point-in-polygon testing. A bounding box pre-filter in SQL reduces the candidate set before the spatial join.

### Data Model

Field ordering in the SQLAlchemy model matches the assignment specification. The `api` field serves as the primary key. Timestamps use `datetime.now(timezone.utc)` with lambda wrappers to avoid the deprecated `datetime.utcnow()`.

### Error Handling

The scraper implements exponential backoff with 3 retries per API. Failed APIs are tracked and reported but don't halt the batch. The API's `/scrape/start` endpoint runs scraping in a background task with status polling via `/scrape/status`.
