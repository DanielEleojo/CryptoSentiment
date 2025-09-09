"""
Twitter Sentiment Scraper for Cryptocurrency Analysis

This module scrapes Twitter for cryptocurrency-related posts using Playwright
and saved authentication state for sentiment analysis in trading algorithms.
"""

import asyncio
import os
from typing import List, Optional
from urllib.parse import quote
from playwright.async_api import async_playwright


class TwitterSentimentScraper:
    """Twitter scraper for cryptocurrency sentiment analysis."""
    
    DEFAULT_AUTH_FILE = "twitter_auth.json"
    DEFAULT_TIMEOUT = 15000
    DEFAULT_SCROLL_COUNT = 3
    DEFAULT_SCROLL_DISTANCE = 3000
    DEFAULT_WAIT_TIME = 2000
    
    def __init__(self, auth_file: str = DEFAULT_AUTH_FILE):
        """
        Initialize Twitter scraper with authentication file.
        
        Args:
            auth_file (str): Path to authentication state file
        """
        self.auth_file = auth_file
        
    async def scrape_twitter_sentiment(self, 
                                     keyword: str, 
                                     max_tweets: int = 50,
                                     scroll_count: int = DEFAULT_SCROLL_COUNT) -> List[str]:
        """
        Scrape Twitter for tweets containing specified cryptocurrency keywords.
        
        Args:
            keyword (str): Search term (e.g., "#bitcoin", "ethereum")
            max_tweets (int): Maximum number of tweets to collect
            scroll_count (int): Number of times to scroll for more content
            
        Returns:
            List[str]: List of tweet content strings
        """
        if not os.path.exists(self.auth_file):
            raise FileNotFoundError(f"Authentication file {self.auth_file} not found. "
                                  f"Please run login_state.py first.")
        
        encoded_keyword = quote(keyword)
        search_url = (f"https://twitter.com/search?q={encoded_keyword}"
                     f"&src=typed_query&f=live")
        tweets = []
        
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=False)
            context = await browser.new_context(storage_state=self.auth_file)
            page = await context.new_page()
            
            try:
                await page.goto(search_url)
                print(f"🌐 Navigated to Twitter search for: {keyword}")
                await page.wait_for_timeout(5000)  # Allow page to settle
                
                # Wait for tweets to load
                try:
                    await page.wait_for_selector("article", timeout=self.DEFAULT_TIMEOUT)
                    print("✅ Tweets loaded successfully.")
                except Exception:
                    print("❌ Tweets failed to load. Saving debug screenshot...")
                    await page.screenshot(path="debug_twitter_error.png", full_page=True)
                    return []
                
                # Scroll to load additional tweets
                for scroll_iteration in range(scroll_count):
                    await page.mouse.wheel(0, self.DEFAULT_SCROLL_DISTANCE)
                    await page.wait_for_timeout(self.DEFAULT_WAIT_TIME)
                    print(f"📜 Scroll {scroll_iteration + 1}/{scroll_count} completed")
                
                # Extract tweet content
                tweet_elements = await page.query_selector_all("article")
                for element in tweet_elements:
                    try:
                        tweet_content = await element.inner_text()
                        if tweet_content.strip():  # Only add non-empty tweets
                            tweets.append(tweet_content.strip())
                            
                        if len(tweets) >= max_tweets:
                            break
                            
                    except Exception as extraction_error:
                        print(f"⚠️  Error extracting tweet: {extraction_error}")
                        continue
                
                print(f"✅ Successfully scraped {len(tweets)} tweets")
                
            except Exception as scraping_error:
                print(f"❌ Error during Twitter scraping: {scraping_error}")
                
            finally:
                await browser.close()
                
        return tweets[:max_tweets]  # Ensure we don't exceed max_tweets
    
    def display_tweets(self, tweets: List[str], max_display: int = 10) -> None:
        """
        Display scraped tweets in a formatted manner.
        
        Args:
            tweets (List[str]): List of tweet content
            max_display (int): Maximum number of tweets to display
        """
        if not tweets:
            print("❌ No tweets to display")
            return
            
        print(f"\n📊 Displaying {min(len(tweets), max_display)} of {len(tweets)} tweets:\n")
        
        for index, tweet in enumerate(tweets[:max_display], 1):
            # Truncate long tweets for display
            display_content = tweet[:250] + "..." if len(tweet) > 250 else tweet
            print(f"{index:2d}. {display_content}")
            print("-" * 80)


async def main():
    """Main execution function for Twitter sentiment scraping."""
    # Configuration
    SEARCH_KEYWORDS = ["#bitcoin", "#ethereum", "#crypto"]
    TWEETS_PER_KEYWORD = 30
    
    scraper = TwitterSentimentScraper()
    
    for keyword in SEARCH_KEYWORDS:
        print(f"\n🔍 Scraping Twitter for: {keyword}")
        tweets = await scraper.scrape_twitter_sentiment(
            keyword=keyword, 
            max_tweets=TWEETS_PER_KEYWORD
        )
        
        if tweets:
            scraper.display_tweets(tweets, max_display=5)
        else:
            print(f"❌ No tweets found for {keyword}")
        
        # Small delay between different keyword searches
        await asyncio.sleep(2)


if __name__ == "__main__":
    # Set default search term for direct execution
    SEARCH_TERM = "#bitcoin"
    
    async def single_search():
        scraper = TwitterSentimentScraper()
        results = await scraper.scrape_twitter_sentiment(SEARCH_TERM, max_tweets=50)
        scraper.display_tweets(results, max_display=10)
    
    asyncio.run(single_search())
