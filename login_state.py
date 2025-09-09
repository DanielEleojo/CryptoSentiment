"""
Twitter Authentication State Manager

This module handles saving Twitter login state using Playwright for automated
social media sentiment analysis in cryptocurrency trading.
"""

import asyncio
import os
from playwright.async_api import async_playwright


async def save_login_state(timeout_seconds: int = 30000, 
                          auth_file: str = "twitter_auth.json",
                          login_url: str = "https://twitter.com/login") -> None:
    """
    Save Twitter login state to a JSON file for future automated sessions.
    
    Args:
        timeout_seconds (int): Time to wait for manual login (default: 30 seconds)
        auth_file (str): Output file for authentication state
        login_url (str): Twitter login URL
        
    Returns:
        None
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(login_url)
            print("🔐 Please log in manually...")
            
            # Wait for user to complete manual login
            await page.wait_for_timeout(timeout_seconds)
            
            # Save authentication state including cookies and local storage
            await context.storage_state(path=auth_file)
            print(f"✅ Login state saved to {auth_file}")
            
        except Exception as error:
            print(f"❌ Error during login process: {error}")
            
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(save_login_state())
