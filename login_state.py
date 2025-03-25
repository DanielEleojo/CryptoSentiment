# save_login_state.py
import asyncio
from playwright.async_api import async_playwright

async def save_login():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Show browser
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://twitter.com/login")

        print("🔐 Please log in manually...")

        # Wait for user to log in manually
        await page.wait_for_timeout(30000)  # Wait 30 seconds

        # Save cookies + local storage to a file
        await context.storage_state(path="twitter_auth.json")
        print("✅ Login state saved to twitter_auth.json")

        await browser.close()

asyncio.run(save_login())
