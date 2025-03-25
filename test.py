import asyncio
from playwright.async_api import async_playwright
from urllib.parse import quote  # For proper URL encoding

SEARCH_TERM = "#bitcoin"  # You can try just 'bitcoin' too

async def scrape_twitter(keyword):
    encoded_keyword = quote(keyword)  # Makes #bitcoin => %23bitcoin
    url = f"https://twitter.com/search?q={encoded_keyword}&src=typed_query&f=live"
    tweets = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state="twitter_auth.json")  # use saved login
        page = await context.new_page()
        await page.goto(url)
        
        print("🌐 Navigated to Twitter search URL...")
        await page.wait_for_timeout(5000)  # Let everything settle

        try:
            await page.wait_for_selector("article", timeout=15000)
            print("✅ Tweets loaded.")
        except:
            print("❌ Tweets didn't load. Saving screenshot...")
            await page.screenshot(path="debug_twitter.png", full_page=True)
            await browser.close()
            return []

        # Scroll to load more tweets
        for _ in range(3):
            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(2000)

        tweet_elements = await page.query_selector_all("article")
        for tweet in tweet_elements:
            try:
                content = await tweet.inner_text()
                tweets.append(content)
            except:
                continue

        await browser.close()
        return tweets

if __name__ == "__main__":
    results = asyncio.run(scrape_twitter(SEARCH_TERM))
    print(f"\n🧠 Scraped {len(results)} tweets for '{SEARCH_TERM}':\n")
    for i, tweet in enumerate(results[:10], 1):
        print(f"{i}. {tweet[:250]}...\n{'-'*40}")
