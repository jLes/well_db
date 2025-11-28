#!/usr/bin/env python3
"""
Extract responses from NM OCD site using BeautifulSoup and Playwright to compare 

Usage:

    python utils/compare_scrapers [API_NUMBER]
"""
import sys
from pathlib import Path
import asyncio

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

BASE_URL = "https://wwwapps.emnrd.nm.gov/OCD/OCDPermitting/Data/WellDetails.aspx"
OUTPUT_DIR = Path(__file__).parent / "scraper_comparisons"

def scrape_with_beautifulsoup(url: str) -> str:
    """Fetch using requests and BeautifulSoup"""
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator='\n', strip=True)

async def scrape_with_playwright(url: str) -> str:
    """Fetch using Playwright"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(1)
        text = await page.evaluate('() => document.body.innerText')
        await browser.close()
    return text

async def main():
    api = sys.argv[1] if len(sys.argv) > 1 else "30-015-25330"
    print(f"Fetching API: {api}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    url = f"{BASE_URL}?api={api}"
    print(f"Target URL: {url}\n")

    print("[1/2] BeautifulSoup . . .")
    bs_text = scrape_with_beautifulsoup(url)
    bs_file = OUTPUT_DIR / f"{api.replace('-','_')}_beautifulsoup.txt"
    bs_file.write_text(bs_text)
    print(f". . . . . . {len(bs_text):,} bytes -> {bs_file.name}")

    print("[2/2] Playwright . . .")
    pw_text = await scrape_with_playwright(url)
    pw_file = OUTPUT_DIR / f"{api.replace('-',"_")}_playwright.txt"
    pw_file.write_text(pw_text)
    print(f". . . . . .{len(pw_text):,} bytes -> {pw_file.name}")

    print(f"\nOutput: {OUTPUT_DIR}/")

if __name__ == "__main__":
    asyncio.run(main())
