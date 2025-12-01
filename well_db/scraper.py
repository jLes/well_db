"""
Web scraper for New Mexico Oil Conservation Division well data.
Uses Playwright to handle server-side rendered content.
"""
import csv
import random
import re
import logging
import asyncio
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass

from playwright.async_api import (
    async_playwright,
    Browser,
    TimeoutError as PlaywrightTimeout,
    Error as PlaywrightError,
)

#Instead of hardcoding field mappings here, I define them once in models.py and re-use where able
from well_db.models import WEBPAGE_FIELD_MAPPINGS, DEFAULT_CSV_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)

# Constants
BASE_URL = "https://wwwapps.emnrd.nm.gov/OCD/OCDPermitting/Data/WellDetails.aspx"
DEFAULT_TIMEOUT = 25000  # 25 seconds
MAX_RETRIES = 3  # Max retries per well on failure
RETRY_BACKOFF = 3  # Base seconds for exponential backoff
DEFAULT_CONCURRENCY = 3  # Default number of concurrent requests (conservative)
REQUEST_DELAY = 1.5  # Minimum seconds between requests per worker


@dataclass
class ScrapeResult:
    """Result of a batch scrape operation."""
    success_count: int
    fail_count: int
    failed_apis: list[str]

class WellScraper:
    """Scraper for NM OCD well data using Playwright."""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser: Optional[Browser] = None

    async def __aenter__(self):
        """Async context manager entry."""
        playwright = await async_playwright().start()
        # Only using one browser instance for simplicity
        self.browser = await playwright.chromium.launch(headless=self.headless)
        self._playwright = playwright
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
        await self._playwright.stop()

    async def scrape_well(self, api_number: str, retry_count: int = 0) -> Optional[dict]:
        """
        Scrape data for a single well with retry logic.

        Args:
            api_number: Well API number (e.g., "30-015-25325")
            retry_count: Current retry attempt (internal use)

        Returns:
            Dictionary of well data or None if scraping failed
        """
        if not self.browser:
            raise RuntimeError("Scraper not initialized. Use 'async with' context manager.")

        context = await self.browser.new_context()
        page = await context.new_page()

        try:
            url = f"{BASE_URL}?api={api_number}"
            logger.debug(f"Fetching: {url}")

            await page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT)

            # Wait for the General Well Information section to render
            # This new approach avoids the fixed delay approach, and having to wait for the full page text.
            try:
                await page.wait_for_selector("#general_information", timeout=DEFAULT_TIMEOUT)
            except PlaywrightTimeout:
                logger.debug(f"Timeout waiting for #general_information for {api_number}")
                if retry_count < MAX_RETRIES:
                    backoff = RETRY_BACKOFF * (2 ** retry_count)
                    logger.debug(f"Retrying {api_number} in {backoff}s (selector timeout)...")
                    await asyncio.sleep(backoff)
                    await page.close()
                    await context.close()
                    return await self.scrape_well(api_number, retry_count + 1)
                logger.error(f"Failed to find content for {api_number} after {MAX_RETRIES} retries")
                return None

            # Extract text from just the General Well Information fieldset
            text_content = await self._extract_well_info_section(page)

            if not text_content:
                logger.debug(f"Empty content extracted for {api_number}")
                if retry_count < MAX_RETRIES:
                    backoff = RETRY_BACKOFF * (2 ** retry_count)
                    logger.debug(f"Retrying {api_number} in {backoff}s (empty content)...")
                    await asyncio.sleep(backoff)
                    await page.close()
                    await context.close()
                    return await self.scrape_well(api_number, retry_count + 1)
                logger.error(f"Failed to get content for {api_number} after {MAX_RETRIES} retries")
                return None

            well_data = self._extract_fields(text_content, api_number)

            return well_data

        except PlaywrightTimeout as e:
            # Timeout errors are good candidates for retry
            logger.debug(f"Timeout scraping {api_number} (attempt {retry_count + 1}): {e}")

            if retry_count < MAX_RETRIES:
                backoff = RETRY_BACKOFF * (2 ** retry_count)
                logger.debug(f"Retrying {api_number} in {backoff}s...")
                await asyncio.sleep(backoff)
                return await self.scrape_well(api_number, retry_count + 1)

            logger.error(f"Failed to scrape {api_number} after {MAX_RETRIES} retries (timeout)")
            return None

        except PlaywrightError as e:
            # Other Playwright errors (navigation failed, browser crashed, etc.)
            logger.debug(f"Playwright error scraping {api_number} (attempt {retry_count + 1}): {e}")

            if retry_count < MAX_RETRIES:
                backoff = RETRY_BACKOFF * (2 ** retry_count)
                logger.debug(f"Retrying {api_number} in {backoff}s...")
                await asyncio.sleep(backoff)
                return await self.scrape_well(api_number, retry_count + 1)

            logger.error(f"Failed to scrape {api_number} after {MAX_RETRIES} retries (playwright error)")
            return None

        except Exception as e:
            # Unexpected errors - log but don't retry (potentially broken request?)
            logger.error(f"Unexpected error scraping {api_number}: {type(e).__name__}: {e}")
            return None

        finally:
            await page.close()
            await context.close()

    async def _extract_well_info_section(self, page) -> str:
        """Extract text content from the General Well Information fieldset only."""
        try:
            # Get the first fieldset.data_container which contains General Well Information
            fieldset = page.locator("fieldset.data_container").first
            text_content = await fieldset.inner_text()
            return text_content or ""
        except Exception as e:
            logger.warning(f"Could not extract well info section: {e}")
            return ""

    def _extract_fields(self, text_content: str, api_number: str) -> dict:
        """Extract all requested fields from the page text."""
        data = {"api": api_number}
        logger.debug(f"Text content size: {len(text_content)} bytes")

        # Extract labeled fields using simple text pattern matching
        for label, field_name in WEBPAGE_FIELD_MAPPINGS.items():
            value = self._extract_label_value(label, text_content)
            data[field_name] = value

        # Special handling for operator
        operator = self._extract_operator(text_content)
        if operator:
            data["operator"] = operator

        # Special handling for surface_location (may span multiple lines)
        surface_loc = self._extract_surface_location(text_content)
        if surface_loc:
            data["surface_location"] = surface_loc

        # Special handling for TVD (True Vertical Depth)
        tvd = self._extract_tvd(text_content)
        if tvd is not None:
            data["tvd"] = tvd

        # Extract coordinates
        lat, lon, crs = self._extract_coordinates(text_content)
        data["latitude"] = lat
        data["longitude"] = lon
        data["crs"] = crs

        return data

    def _extract_operator(self, text_content: str) -> Optional[str]:
        """Extract operator name from text content."""
        try:
            match = re.search(r'Operator:\s*(.+)', text_content)
            if match:
                return match.group(1).strip()
        except Exception as e:
            logger.debug(f"Could not extract operator: {e}")
        return None

    def _extract_surface_location(self, text_content: str) -> Optional[str]:
        """Extract full surface location including FNL/FWL data."""
        try:
            # Look for surface location pattern
            match = re.search(r'Surface Location:\s*(.+)', text_content)
            if match:
                return match.group(1).strip()

            # Alternative: look for the pattern with FNL/FWL
            match = re.search(r'([A-Z]-\d+-\d+[A-Z]-\d+[A-Z]\s+\d+\s+FNL\s+\d+\s+FWL)', text_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        except Exception as e:
            logger.debug(f"Could not extract surface location: {e}")
        return None

    def _extract_tvd(self, text_content: str) -> Optional[str]:
        """Extract True Vertical Depth (TVD) from text content."""
        try:
            patterns = [
                r'True Vertical Depth:\s*([\d,]+)',
                r'TVD:\s*([\d,]+)',
                r'True Vert(?:ical)?[.\s]*Depth:\s*([\d,]+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    return match.group(1).replace(',', '').strip()
        except Exception as e:
            logger.debug(f"Could not extract TVD: {e}")
        return None

    def _extract_label_value(self, label: str, text_content: str) -> Optional[str]:
        """
        Extract value for a given label from plain text content.

        Args:
            label: The label text to search for (e.g., "Operator:")
            text_content: Plain text from page innerText
        """
        try:
            # Pattern: Label: Value (capture to end of line, horizontal whitespace only)
            pattern = rf'{re.escape(label)}[ \t]*(.*)$'
            match = re.search(pattern, text_content, re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        except Exception as e:
            logger.debug(f"Could not extract {label}: {e}")
        return None

    def _extract_coordinates(self, text_content: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
        """Extract latitude, longitude, and CRS from the provided content."""
        lat, lon, crs = None, None, None

        try:
            # Look for coordinate patterns in the content
            # Pattern: latitude, longitude with optional CRS
            coord_pattern = r'(-?\d+\.\d+)\s*,?\s*(-?\d+\.\d+)'
            matches = re.findall(coord_pattern, text_content)

            for match in matches:
                try:
                    potential_lat = float(match[0])
                    potential_lon = float(match[1])

                    # Validate as NM coordinates (rough bounds)
                    if 31 <= potential_lat <= 37 and -109 <= potential_lon <= -103:
                        lat, lon = potential_lat, potential_lon
                        break
                    # Sometimes lon comes first
                    if 31 <= potential_lon <= 37 and -109 <= potential_lat <= -103:
                        lat, lon = potential_lon, potential_lat
                        break
                except ValueError:
                    continue

            # Look for CRS (usually NAD83 or NAD27)
            crs_pattern = r'(NAD\d{2}|WGS\d{2})'
            crs_match = re.search(crs_pattern, text_content)
            if crs_match:
                crs = crs_match.group(1)

        except Exception as e:
            logger.debug(f"Could not extract coordinates: {e}")

        return lat, lon, crs


### Batch scraping 

def load_api_numbers_from_csv(csv_path: Path) -> list[str]:
    """Load all API numbers from the provided CSV file."""
    if not csv_path.exists():
        return []

    api_numbers = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            api_numbers.append(row["api"])
    return api_numbers

async def scrape_from_csv(
    csv_path: Path,
    progress_callback: Optional[callable] = None
) -> list[dict]:
    """
    Scrape well data for all API numbers in a CSV file.

    Args:
        csv_path: Path to CSV file with 'api' vector of known well IDs
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of well data dictionaries
    """
    # Read API numbers from CSV
    api_numbers = load_api_numbers_from_csv(csv_path)

    logger.info(f"Found {len(api_numbers)} API numbers to scrape")

    results = []
    async with WellScraper() as scraper:
        for i, api in enumerate(api_numbers):
            logger.info(f"Scraping {i + 1}/{len(api_numbers)}: {api}")

            well_data = await scraper.scrape_well(api)
            if well_data:
                results.append(well_data)
            else:
                logger.warning(f"Failed to scrape: {api}")

            if progress_callback:
                progress_callback(i + 1, len(api_numbers))

            # Respectful delay with jitter
            await asyncio.sleep(REQUEST_DELAY + random.uniform(0, 0.5))

    logger.info(f"Successfully scraped {len(results)}/{len(api_numbers)} wells")
    return results


async def scrape_batch(
    api_numbers: list[str],
    concurrency: int = DEFAULT_CONCURRENCY,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    on_result: Optional[Callable[[dict], None]] = None,
    on_error: Optional[Callable[[str, str], None]] = None,
) -> ScrapeResult:
    """
    Scrape multiple wells using a worker pool pattern.

    Uses an async queue with N=concurrency workers, each processing one request at a time
    with a delay between requests to avoid overwhelming the server.

    Args:
        api_numbers: List of API numbers to scrape
        concurrency: Number of worker tasks (default: 3)
        on_progress: Callback(api, current, total) called when a scrape completes
        on_result: Callback(well_data) called with successful scrape data
        on_error: Callback(api, error_msg) called on scrape failure

    Returns:
        ScrapeResult with success/fail counts and list of failed APIs
    """
    if not api_numbers:
        return ScrapeResult(success_count=0, fail_count=0, failed_apis=[])

    queue: asyncio.Queue[str] = asyncio.Queue()
    success_count = 0
    fail_count = 0
    failed_apis: list[str] = []
    completed = 0
    total = len(api_numbers)
    lock = asyncio.Lock()  # Protect shared counters

    # Populate the queue
    for api in api_numbers:
        await queue.put(api)

    async def worker(scraper: WellScraper, worker_id: int) -> None:
        """Worker that processes APIs from the queue."""
        nonlocal success_count, fail_count, completed

        while True:
            try:
                # Get next API from queue (non-blocking check if empty)
                api = queue.get_nowait()
            except asyncio.QueueEmpty:
                return  # No more work

            try:
                well_data = await scraper.scrape_well(api)

                async with lock:
                    completed += 1
                    if well_data:
                        success_count += 1
                        if on_result:
                            on_result(well_data)
                    else:
                        fail_count += 1
                        failed_apis.append(api)
                        if on_error:
                            on_error(api, "No data returned")

                    if on_progress:
                        on_progress(api, completed, total)

            except Exception as e:
                async with lock:
                    completed += 1
                    fail_count += 1
                    failed_apis.append(api)
                    if on_error:
                        on_error(api, str(e))
                    if on_progress:
                        on_progress(api, completed, total)

            finally:
                queue.task_done()

            # Delay before next request, to account for any server-side rate limiting
            await asyncio.sleep(REQUEST_DELAY + random.uniform(0, 0.5))

    logger.info(f"Starting batch scrape of {total} APIs with {concurrency} workers")

    async with WellScraper(headless=True) as scraper:
        # Stagger worker starts to avoid initial burst
        workers = []
        for i in range(concurrency):
            await asyncio.sleep(0.5)  # Stagger each worker start by 0.5s
            workers.append(asyncio.create_task(worker(scraper, i)))

        # Wait for all workers to complete
        await asyncio.gather(*workers)

    logger.info(f"Batch scrape complete: {success_count} succeeded, {fail_count} failed")
    return ScrapeResult(
        success_count=success_count,
        fail_count=fail_count,
        failed_apis=failed_apis,
    )


### Constants and helper functions for standalone testing

# API number format: XX-XXX-XXXXX (e.g., 30-015-25327)
API_PATTERN = re.compile(r"^\d{2}-\d{3}-\d{5}$")

def is_valid_api_format(api: str) -> bool:
    """Check if the API number matches the expected format (XX-XXX-XXXXX)."""
    return bool(API_PATTERN.match(api))

# DEFAULT_CSV_PATH is imported from models.py for convenience here . . . 
def get_random_api(csv_path: Path = DEFAULT_CSV_PATH) -> Optional[str]:
    """Get a random API number from the CSV file."""
    api_numbers = load_api_numbers_from_csv(csv_path)
    if not api_numbers:
        return None
    return random.choice(api_numbers)


if __name__ == "__main__":
    import sys

    async def test_single(api_number: str):
        """Test scraping a single well."""
        async with WellScraper(headless=True) as scraper:
            result = await scraper.scrape_well(api_number)
            if result:
                print(f"\nScraped data for {api_number}:")
                print("-" * 50)
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Scraping failed for {api_number}")

    # Get API from command line argument
    arg_api = sys.argv[1] if len(sys.argv) > 1 else None

    # Validate or get random API
    if arg_api and is_valid_api_format(arg_api):
        api = arg_api
        print(f"Using provided API: {api}")
    else:
        if arg_api:
            print(f"Invalid API format: '{arg_api}' (expected XX-XXX-XXXXX)")

        # Try to get a random API from CSV
        api = get_random_api()
        if api:
            print(f"Using random API from CSV: {api}")
        else:
            # Fallback to a known working API
            api = "30-015-25343"
            print(f"CSV not found, using fallback API: {api}")

    asyncio.run(test_single(api))
