"""
Configuration Management for CryptoSentiment Analysis

This module handles secure configuration including API keys and settings.
Use environment variables for production deployment.
"""

import os
from typing import Dict, Any


class Config:
    """Configuration management for cryptocurrency sentiment analysis."""
    
    # API Configuration
    COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY')
    
    # Reddit API Configuration  
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    
    # Default API Settings
    DEFAULT_CRYPTO_LIMIT = 10
    DEFAULT_REDDIT_POST_LIMIT = 200
    DEFAULT_REQUEST_TIMEOUT = 30
    
    # Social Media Settings
    TWITTER_AUTH_FILE = "twitter_auth.json"
    DEFAULT_SUBREDDITS = ['cryptocurrency', 'bitcoin', 'ethereum', 'crypto']
    
    # Model Settings
    SENTIMENT_THRESHOLD_POSITIVE = 0.1
    SENTIMENT_THRESHOLD_NEGATIVE = -0.1
    
    @classmethod
    def get_api_settings(cls) -> Dict[str, Any]:
        """
        Get API configuration settings.
        
        Returns:
            Dict[str, Any]: API configuration parameters
        """
        return {
            'coinmarketcap_api_key': cls.COINMARKETCAP_API_KEY,
            'default_crypto_limit': cls.DEFAULT_CRYPTO_LIMIT,
            'default_reddit_limit': cls.DEFAULT_REDDIT_POST_LIMIT,
            'request_timeout': cls.DEFAULT_REQUEST_TIMEOUT
        }
    
    @classmethod
    def validate_configuration(cls) -> bool:
        """
        Validate that required configuration is available.
        
        Returns:
            bool: True if configuration is valid
        """
        if not cls.COINMARKETCAP_API_KEY:
            print("⚠️  Warning: COINMARKETCAP_API_KEY not set in environment")
            return False
        
        return True


# Load configuration on import
config = Config()