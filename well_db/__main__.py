#!/usr/bin/env python3
"""Entrypoint for the well_db package."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent.resolve()


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


def create_parser() -> argparse.ArgumentParser:
    """Create an argument parser with subcommands for poulating the database. Eventually can provide access to spooling up the API server, and maybe a Docker runtime."""
    parser = argparse.ArgumentParser(
        prog="well_db",
        description="WellDB - Scrape and serve New Mexico oil and gas well data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  well_db scrape                  Scrape wells using default CSV
  well_db scrape custom_apis.csv  Scrape wells from custom CSV
  well_db scrape --missing        Scrape only wells not already in database
  well_db scrape --force          Re-scrape all wells (upserts existing)
  well_db scrape -c 3             Scrape with 5 concurrent requests
  well_db delete                  Delete the database file (with prompt)
  well_db delete -y               Delete the database file (no prompt)
        """,
          )
    
    # Anticipating adding subcommands for things like "serve" and others, so need subparsers
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command")

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

    return parser

def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    commands = {
        "scrape": cmd_scrape,
        "delete": cmd_delete,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())