"""
Cryptocurrency Sentiment Analysis Module

This module provides comprehensive cryptocurrency sentiment analysis by combining:
1. Real-time market data from CoinMarketCap API
2. Social media sentiment from Reddit and Twitter
3. VADER sentiment analysis for text processing
4. Machine learning models for price prediction

Author: Quantitative Finance Analyst
Purpose: Cryptocurrency trading algorithm with sentiment-driven predictions
"""

import os
import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


class CryptoMarketDataFetcher:
    """Handles cryptocurrency market data retrieval from CoinMarketCap API."""
    
    BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    DEFAULT_CONVERT_CURRENCY = "USD"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the market data fetcher.
        
        Args:
            api_key (str, optional): CoinMarketCap API key. 
                                   If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('COINMARKETCAP_API_KEY')
        if not self.api_key:
            raise ValueError("CoinMarketCap API key is required. Set via parameter or "
                           "COINMARKETCAP_API_KEY environment variable.")
    
    def get_crypto_data(self, 
                       start_rank: int = 1, 
                       limit: int = 5, 
                       convert_currency: str = DEFAULT_CONVERT_CURRENCY) -> pd.DataFrame:
        """
        Fetch cryptocurrency listings from CoinMarketCap API.
        
        Args:
            start_rank (int): Starting rank of cryptocurrencies (1 = Bitcoin)
            limit (int): Number of cryptocurrencies to retrieve
            convert_currency (str): Target currency for price conversion
            
        Returns:
            pd.DataFrame: Cryptocurrency market data with price and volume information
            
        Raises:
            requests.RequestException: If API request fails
            ValueError: If API returns invalid data
        """
        request_parameters = {
            'start': str(start_rank),
            'limit': str(limit),
            'convert': convert_currency
        }
        
        request_headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key
        }
        
        try:
            response = requests.get(
                self.BASE_URL, 
                headers=request_headers, 
                params=request_parameters,
                timeout=30
            )
            response.raise_for_status()
            
            api_data = response.json()
            
            if 'data' not in api_data:
                raise ValueError("Invalid API response format")
            
            crypto_listings = api_data['data']
            crypto_df = pd.DataFrame(crypto_listings)
            
            # Extract and normalize nested quote data
            quote_df = crypto_df['quote'].apply(lambda x: x[convert_currency])
            quote_normalized = pd.json_normalize(quote_df)
            
            # Remove redundant last_updated column from quote data
            if 'last_updated' in quote_normalized.columns:
                quote_normalized = quote_normalized.drop(columns=['last_updated'])
            
            # Combine main data with quote data
            combined_df = pd.concat([
                crypto_df.drop(columns=['quote']), 
                quote_normalized
            ], axis=1)
            
            # Convert timestamp to datetime
            combined_df['last_updated'] = pd.to_datetime(combined_df['last_updated'])
            
            return combined_df
            
        except requests.RequestException as request_error:
            raise requests.RequestException(f"API request failed: {request_error}")
        except (KeyError, ValueError) as data_error:
            raise ValueError(f"Error processing API data: {data_error}")


class RedditSentimentFetcher:
    """Handles Reddit post retrieval for sentiment analysis."""
    
    PUSHSHIFT_BASE_URL = "https://api.pushshift.io/reddit/search/submission/"
    DEFAULT_POST_LIMIT = 100
    
    @staticmethod
    def get_reddit_posts(subreddit: str, 
                        post_limit: int = DEFAULT_POST_LIMIT,
                        before_timestamp: Optional[int] = None,
                        after_timestamp: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch Reddit posts from specified subreddit using Pushshift API.
        
        This method provides access to historical Reddit data without requiring
        separate Reddit API authentication.
        
        Args:
            subreddit (str): Target subreddit name (e.g., 'cryptocurrency')
            post_limit (int): Maximum number of posts to retrieve
            before_timestamp (int, optional): Unix timestamp for posts before this time
            after_timestamp (int, optional): Unix timestamp for posts after this time
            
        Returns:
            pd.DataFrame: Reddit posts with timestamps and combined text content
        """
        api_parameters = {
            'subreddit': subreddit,
            'size': post_limit,
            'sort': 'desc',
            'sort_type': 'created_utc'
        }
        
        if before_timestamp:
            api_parameters['before'] = before_timestamp
        if after_timestamp:
            api_parameters['after'] = after_timestamp
        
        try:
            response = requests.get(
                RedditSentimentFetcher.PUSHSHIFT_BASE_URL, 
                params=api_parameters,
                timeout=30
            )
            response.raise_for_status()
            
            response_data = response.json()
            reddit_posts = response_data.get('data', [])
            
            if not reddit_posts:
                print("⚠️  No Reddit posts retrieved. API response:")
                print(response_data)
                return pd.DataFrame(columns=['created_utc', 'text'])
            
            # Process posts and combine title with content
            processed_posts = []
            for post in reddit_posts:
                post_timestamp = post.get('created_utc')
                post_title = post.get('title', '')
                post_content = post.get('selftext', '')
                combined_text = f"{post_title} {post_content}".strip()
                
                processed_posts.append({
                    'created_utc': post_timestamp,
                    'text': combined_text
                })
            
            posts_df = pd.DataFrame(processed_posts)
            
            if posts_df.empty:
                print("⚠️  No posts with valid timestamps found")
                return posts_df
            
            # Convert Unix timestamps to datetime objects
            posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')
            
            return posts_df
            
        except requests.RequestException as request_error:
            print(f"❌ Reddit API request failed: {request_error}")
            return pd.DataFrame(columns=['created_utc', 'text'])
        except Exception as processing_error:
            print(f"❌ Error processing Reddit data: {processing_error}")
            return pd.DataFrame(columns=['created_utc', 'text'])


class SentimentAnalyzer:
    """Handles sentiment analysis using VADER sentiment intensity analyzer."""
    
    def __init__(self):
        """Initialize VADER sentiment analyzer with required NLTK data."""
        try:
            # Download required NLTK data if not present
            nltk.download('vader_lexicon', quiet=True)
            self.analyzer = SentimentIntensityAnalyzer()
        except Exception as nltk_error:
            raise RuntimeError(f"Failed to initialize NLTK components: {nltk_error}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of given text using VADER.
        
        Args:
            text (str): Text content to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores (negative, neutral, positive, compound)
        """
        if not text or not isinstance(text, str):
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
        return self.analyzer.polarity_scores(text)
    
    def batch_sentiment_analysis(self, texts: List[str]) -> pd.DataFrame:
        """
        Perform sentiment analysis on multiple texts.
        
        Args:
            texts (List[str]): List of text strings to analyze
            
        Returns:
            pd.DataFrame: Sentiment scores for all texts
        """
        sentiment_results = []
        
        for text in texts:
            sentiment_scores = self.analyze_sentiment(text)
            sentiment_results.append(sentiment_scores)
        
        return pd.DataFrame(sentiment_results)


class CryptoPricePredictionModel:
    """Machine learning model for cryptocurrency price prediction using sentiment data."""
    
    def __init__(self):
        """Initialize linear regression model for price prediction."""
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_columns = None
    
    def prepare_features(self, 
                        market_data: pd.DataFrame, 
                        sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine market data with sentiment scores for model training.
        
        Args:
            market_data (pd.DataFrame): Cryptocurrency market data
            sentiment_data (pd.DataFrame): Sentiment analysis results
            
        Returns:
            pd.DataFrame: Combined feature set for machine learning
        """
        # Calculate sentiment aggregates
        sentiment_summary = {
            'avg_sentiment': sentiment_data['compound'].mean(),
            'sentiment_volatility': sentiment_data['compound'].std(),
            'positive_ratio': sentiment_data['pos'].mean(),
            'negative_ratio': sentiment_data['neg'].mean()
        }
        
        # Select relevant market features
        market_features = market_data[[
            'price', 'percent_change_24h', 'percent_change_7d', 
            'market_cap', 'volume_24h'
        ]].copy()
        
        # Add sentiment features
        for sentiment_feature, value in sentiment_summary.items():
            market_features[sentiment_feature] = value
        
        return market_features
    
    def train_model(self, 
                   features: pd.DataFrame, 
                   target: pd.Series) -> Dict[str, float]:
        """
        Train the price prediction model.
        
        Args:
            features (pd.DataFrame): Input features for training
            target (pd.Series): Target values (future prices)
            
        Returns:
            Dict[str, float]: Model performance metrics
        """
        # Handle missing values
        features_clean = features.fillna(features.mean())
        target_clean = target.fillna(target.mean())
        
        # Train the model
        self.model.fit(features_clean, target_clean)
        self.is_trained = True
        self.feature_columns = features_clean.columns.tolist()
        
        # Calculate performance metrics
        predictions = self.model.predict(features_clean)
        
        performance_metrics = {
            'r2_score': r2_score(target_clean, predictions),
            'mean_absolute_error': mean_absolute_error(target_clean, predictions),
            'training_samples': len(features_clean)
        }
        
        return performance_metrics
    
    def predict_price(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict cryptocurrency prices using trained model.
        
        Args:
            features (pd.DataFrame): Input features for prediction
            
        Returns:
            np.ndarray: Predicted prices
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        features_clean = features.fillna(features.mean())
        return self.model.predict(features_clean)


def create_sample_analysis():
    """
    Demonstrate the cryptocurrency sentiment analysis pipeline.
    
    This function showcases the complete workflow from data collection
    to sentiment analysis and price prediction.
    """
    print("🚀 Starting Cryptocurrency Sentiment Analysis Pipeline")
    print("=" * 60)
    
    # Note: In production, API key should come from environment variables
    api_key = os.getenv('COINMARKETCAP_API_KEY', 'c936ec13-bf84-4335-b1df-aa936ab8d2b5')
    
    try:
        # Initialize components
        market_fetcher = CryptoMarketDataFetcher(api_key)
        sentiment_analyzer = SentimentAnalyzer()
        
        # Fetch market data
        print("📊 Fetching cryptocurrency market data...")
        crypto_data = market_fetcher.get_crypto_data(start_rank=1, limit=5)
        print(f"✅ Retrieved data for {len(crypto_data)} cryptocurrencies")
        print(crypto_data[['name', 'symbol', 'price', 'percent_change_24h']].head())
        
        # Fetch Reddit sentiment data
        print("\n📱 Fetching Reddit sentiment data...")
        reddit_posts = RedditSentimentFetcher.get_reddit_posts(
            subreddit='cryptocurrency', 
            post_limit=100
        )
        
        if not reddit_posts.empty:
            print(f"✅ Retrieved {len(reddit_posts)} Reddit posts")
            
            # Perform sentiment analysis
            print("\n🧠 Analyzing sentiment...")
            sentiment_scores = sentiment_analyzer.batch_sentiment_analysis(
                reddit_posts['text'].tolist()
            )
            
            print("✅ Sentiment analysis completed")
            print(f"Average sentiment: {sentiment_scores['compound'].mean():.3f}")
            print(f"Positive posts: {(sentiment_scores['compound'] > 0).sum()}")
            print(f"Negative posts: {(sentiment_scores['compound'] < 0).sum()}")
        else:
            print("⚠️  No Reddit data available for sentiment analysis")
    
    except Exception as analysis_error:
        print(f"❌ Analysis failed: {analysis_error}")


if __name__ == "__main__":
    create_sample_analysis()