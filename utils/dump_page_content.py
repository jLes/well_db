#!/usr/bin/env python3
"""
Helper utility to write random well API(s) page contents to txt.

This will inform the field extraction methods and hopefully narrow the search scope

Usage:
    python utils/dump_page_content.py [options]

Options:
    --csv PATH      Path to the provided CVS with known 'API' well numbers
    --count N       Number of random wells to fetch
    --kb N          Max text size (KB) of output
    --output DIR    Output directory (default: ./page_contents )
    --api API       Specific API number to fetch info for a single well, if exists
"""
import sys
import argparse
import csv
import asyncio
import random
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright

BASE_URL = "https://wwwapps.emnrd.nm.gov/OCD/OCDPermitting/Data/WellDetails.aspx"


async def fetch_page_content(api: str) -> str:
    """
    Fetch page content for a single API with Playwright.
    
    Returns:
        text content from page innerText
    """
    url = f"{BASE_URL}?api={api}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(1)

        text_content = await page.evaluate('() => document.body.innerText')
        await browser.close()

    return text_content

def load_api_numbers(csv_path: Path) -> list[str]:
    """Loads all of the API numbers from the provided CSV"""
    api_numbers = []
    # The provided CSV has a BOM, so:
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            api_numbers.append(row['api'])
    return api_numbers


def sanitize_filename(api: str) -> str:
    """Convert API number to spref filename."""
    return api.replace('-', '_')


async def main():
    parser = argparse.ArgumentParser(description="Dump all page content for random APIs")
    parser.add_argument('--csv', type=Path,
                        default=Path(__file__).parent.parent / 'resources' / 'apis_pythondev_test.csv',
                        help='Path to CSV file with APIs')
    parser.add_argument('--count', type=int, default=3,
                        help='Number of random APIs to fetch')
    parser.add_argument('--kb', type=int, default=20,
                        help='Max KB to dump per file')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'page_contents',
                        help='Output directory')
    parser.add_argument('--api', type=str, default=None,
                        help='Specific API number to fetch')

    args = parser.parse_args()

    print(args.csv)
    csv_exists = Path.is_file(args.csv)

    print(f"CSV file {"exists" if csv_exists else "does not exist"}")

    if not csv_exists:
        return
    
    all_apis = load_api_numbers(args.csv)

    # make output loc
    args.output.mkdir(parents=True, exist_ok=True)
    
    # get API nums to fetch
    if args.api:
        apis_to_fetch = [args.api if args.api in all_apis else ""]
    else:
        apis_to_fetch = random.sample(all_apis, min(args.count, len(all_apis)))
        print("APIs to fetch: ", apis_to_fetch)

    print(f"Fetching {len(apis_to_fetch)} API(s)...")
    print(f"Output directory: {args.output}")
    print(f"Max size per file: {args.kb} KB \n")

    max_bytes = args.kb * 1024

    for api in apis_to_fetch:
        print(f"Fetching {api} . . .")

        if not api:
            print("Invalid API number . . . Skipping")
            continue
        try:
            text_content = await fetch_page_content(api)

            text_truncated = text_content[:max_bytes]

            # Write text file (innerText - easier to read)
            text_file = args.output / f"{sanitize_filename(api)}_text.txt"
            with open(text_file, 'w') as f:
                f.write(f"# API: {api}\n")
                f.write(f"# Fetched: {datetime.now().isoformat()}\n")
                f.write(f"# Total size: {len(text_content):,} bytes\n")
                f.write(f"# Showing: {len(text_truncated):,} bytes\n")
                f.write("#" + "=" * 79 + "\n\n")
                f.write(text_truncated)

            print(f". . . -> {text_file.name} ({len(text_truncated):,} bytes)")
        except Exception as e:
            print(f". . . ERROR: {e}")

        print()

    print(f"Done! Valid API files saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())

