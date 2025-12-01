#!/usr/bin/env python3
"""Entrypoint for the well_db package.

This module provides a simple command-line interface for:
  - Starting the REST API server
  - Running the database scraper
  - Querying wells within a geopolygon
  - Starting Docker containers if host-side python env is not desired

Usage:
    python -m well_db serve              # Start the API server
    python -m well_db scrape <csv_file>  # Scrape wells and populate DB
    python -m well_db polygon <coords>   # Query polygon, save to CSV
    python -m well_db docker [up|down]   # Manage Docker containers

Examples:
    uv run python -m well_db serve --port 8080
    uv run python -m well_db scrape ./resources/apis_pythondev_test.csv
    uv run python -m well_db polygon test -o results.csv ## Use test polygon from assignment
    uv run python -m well_db delete --yes
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent.resolve()

def cmd_serve(args: Namespace) -> int:
    """Start the FastAPI server."""
    import uvicorn

    from well_db.api import app
    from well_db.database import init_db

    # Ensure database is initialized
    init_db()

    print(f"Starting WellDB API server at http://{args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    print(f"Database status:   http://{args.host}:{args.port}/db/status")

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_scrape(args: Namespace) -> int:
    """Scrape wells from CSV and load into database."""
    from tqdm import tqdm

    from well_db.database import init_db, db_session, upsert_well, count_wells, get_all_apis, get_db_session
    from well_db.models import DEFAULT_CSV_PATH
    from well_db.scraper import load_api_numbers_from_csv, scrape_batch

    # Use provided path or fall back to default
    csv_path = Path(args.csv_file) if args.csv_file else DEFAULT_CSV_PATH
    if not csv_path.exists():
        # Try default if provided path doesn't exist
        if args.csv_file and DEFAULT_CSV_PATH.exists():
            print(f"Warning: {csv_path} not found, using default: {DEFAULT_CSV_PATH}")
            csv_path = DEFAULT_CSV_PATH
        else:
            print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
            return 1

    # Load API numbers from CSV using shared function
    api_numbers = load_api_numbers_from_csv(csv_path)

    if not api_numbers:
        print("Error: No API numbers found in CSV", file=sys.stderr)
        return 1

    print(f"Found {len(api_numbers)} API numbers to scrape")

    # Initialize database
    init_db()

    # Check for existing data in database
    db = get_db_session()
    existing_count = count_wells(db)
    existing_apis = set(get_all_apis(db))
    db.close()

    if existing_count > 0 and not args.missing and not args.force:
        print(f"Error: Database already contains {existing_count} wells.", file=sys.stderr)
        print("Use --missing to only scrape new APIs, or --force to re-scrape all.", file=sys.stderr)
        return 1

    # If --missing flag, only scrape APIs not in database
    if args.missing:
        original_count = len(api_numbers)
        api_numbers = [api for api in api_numbers if api not in existing_apis]
        print(f"Skipping {original_count - len(api_numbers)} already in database")

        if not api_numbers:
            print("Database is complete. Nothing to scrape.")
            return 0

    # Set up progress bar and callbacks
    pbar = tqdm(total=len(api_numbers), desc="Scraping", unit="well")

    def on_progress(api: str, current: int, total: int) -> None:
        pbar.update(1)
        pbar.set_postfix_str(api)

    def on_result(well_data: dict) -> None:
        with db_session() as db:
            upsert_well(db, well_data)

    def on_error(api: str, error_msg: str) -> None:
        # Only log final failures, not retries (those are handled silently)
        pass

    # Run the async scraper
    print(f"Scraping with concurrency={args.concurrency}")
    result = asyncio.run(scrape_batch(
        api_numbers=api_numbers,
        concurrency=args.concurrency,
        on_progress=on_progress,
        on_result=on_result,
        on_error=on_error,
    ))

    pbar.close()
    print(f"\nScraping complete: {result.success_count} succeeded, {result.fail_count} failed")

    if result.failed_apis:
        print(f"Failed APIs: {', '.join(result.failed_apis[:10])}")
        if len(result.failed_apis) > 10:
            print(f"  ... and {len(result.failed_apis) - 10} more")

    return 0 if result.fail_count == 0 else 1


def cmd_delete(args: Namespace) -> int:
    """Delete the database file."""
    from well_db.database import DB_PATH

    if not DB_PATH.exists():
        print(f"Database file not found: {DB_PATH}")
        return 0

    if not args.yes:
        response = input(f"Delete database at {DB_PATH}? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    DB_PATH.unlink()
    print(f"Deleted: {DB_PATH}")
    return 0


def cmd_polygon(args: Namespace) -> int:
    """Query wells within a polygon and save results to CSV."""
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    from well_db.database import get_db_session
    from well_db.models import WellData

    # Parse coordinates
    if args.coords == "test":
        # Test polygon from the assignment PDF
        points = [
            (32.81, -104.19),
            (32.66, -104.32),
            (32.54, -104.24),
            (32.50, -104.03),
            (32.73, -104.01),
            (32.79, -103.91),
            (32.84, -104.05),
            (32.81, -104.19),
        ]
        print("Using test polygon from assignment")
    else:
        import re
        try:
            # Parse format: [(lat,lon),(lat,lon),...]
            pattern = r'\(([^)]+)\)'
            matches = re.findall(pattern, args.coords)

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
                print("Error: Polygon must have at least 3 points", file=sys.stderr)
                return 1

        except ValueError as e:
            print(f"Error parsing coordinates: {e}", file=sys.stderr)
            print("Expected format: [(lat1,lon1),(lat2,lon2),(lat3,lon3),...]")
            return 1

    db = get_db_session()

    # Bounding box pre-filter for performance
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    candidates = db.query(WellData).filter(
        WellData.latitude.isnot(None),
        WellData.longitude.isnot(None),
        WellData.latitude.between(min_lat, max_lat),
        WellData.longitude.between(min_lon, max_lon),
    ).all()

    db.close()

    if not candidates:
        print("No wells found in bounding box")
        return 0

    # Create polygon in proper GIS format: (lon, lat) for x, y
    # Input is (lat, lon), so swap for Shapely/GeoPandas
    polygon_coords_xy = [(lon, lat) for lat, lon in points]
    search_polygon = Polygon(polygon_coords_xy)

    # Create GeoDataFrame for polygon with WGS84 CRS
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
    matches = gpd.sjoin(wells_gdf, polygon_gdf, predicate="within")
    matching_apis = sorted(matches["api"].tolist())

    # Write results to CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["api"])
        for api in matching_apis:
            writer.writerow([api])

    print(f"Found {len(matching_apis)} wells in polygon")
    print(f"Results saved to: {output_path}")
    return 0


def cmd_docker(args: Namespace) -> int:
    """Manage Docker containers."""
    project_root = get_project_root()
    compose_file = project_root / "docker-compose.yml"

    if not compose_file.exists():
        print(f"Error: docker-compose.yml not found at {compose_file}", file=sys.stderr)
        return 1

    action = args.action

    if action == "up":
        print("Starting Docker containers...")
        cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
        if args.build:
            cmd.append("--build")

    elif action == "down":
        print("Stopping Docker containers...")
        cmd = ["docker", "compose", "-f", str(compose_file), "down"]

    elif action == "logs":
        cmd = ["docker", "compose", "-f", str(compose_file), "logs", "-f"]

    elif action == "scrape":
        print("Running scraper in Docker...")
        cmd = ["docker", "compose", "-f", str(compose_file), "run", "--rm", "scrape"]

    elif action == "status":
        cmd = ["docker", "compose", "-f", str(compose_file), "ps"]

    else:
        print(f"Unknown docker action: {action}", file=sys.stderr)
        return 1

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: Docker is not installed or not in PATH", file=sys.stderr)
        return 1
    

def create_parser() -> argparse.ArgumentParser:
    """Create an argument parser with subcommands for populating the database. Eventually can provide access to spooling up the API server, and maybe a Docker runtime."""
    parser = argparse.ArgumentParser(
        prog="well_db",
        description="WellDB - Scrape and serve New Mexico oil and gas well data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  well_db serve                   Start the API server
  well_db serve --port 8080       Start on custom port
  well_db scrape                  Scrape wells using default CSV
  well_db scrape --missing        Scrape only wells not already in database
  well_db scrape -c 3             Scrape with 3 concurrent requests
  well_db polygon test            Query test polygon, save to CSV
  well_db delete -y               Delete the database file
  well_db docker up               Start Docker containers
        """,
          )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command")

    # Serve subcommand
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FastAPI server",
        description="Start the REST API server for querying well data.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    # Scrape subcommand
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Scrape well data and store in SQLite database",
        description="Scrape APIs for each well listed in CSV from NM OCD site.",
    )
    scrape_parser.add_argument(
        "csv_file",
        nargs="?",
        default=None,
        help="Path to CSV file containing API numbers (default: resources/apis_pythondev_test.csv)",
    )
    scrape_parser.add_argument(
        "--missing",
        action="store_true",
        help="Only scrape APIs not already in database",
    )
    scrape_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scrape even if database contains data (will upsert)",
    )
    scrape_parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=3,
        help="Number of concurrent scrape requests (default: 3)",
    )

    # Delete subcommand
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete the database file",
        description="Remove the SQLite database file entirely.",
    )
    delete_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # Polygon subcommand
    polygon_parser = subparsers.add_parser(
        "polygon",
        help="Query wells within a polygon",
        description="Find wells within a polygon and save results to CSV.",
    )
    polygon_parser.add_argument(
        "coords",
        help="Polygon coordinates as '[(lat1,lon1),(lat2,lon2),...]' or 'test' for assignment polygon",
    )
    polygon_parser.add_argument(
        "--output", "-o",
        default="polygon_results.csv",
        help="Output CSV file path (default: polygon_results.csv)",
    )

    # Docker subcommand
    docker_parser = subparsers.add_parser(
        "docker",
        help="Manage Docker containers",
        description="Start, stop, or manage Docker containers for WellDB.",
    )
    docker_parser.add_argument(
        "action",
        choices=["up", "down", "logs", "scrape", "status"],
        help="Docker action to perform",
    )
    docker_parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild images before starting (only for 'up')",
    )

    return parser

def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    commands = {
        "serve": cmd_serve,
        "scrape": cmd_scrape,
        "delete": cmd_delete,
        "polygon": cmd_polygon,
        "docker": cmd_docker,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())