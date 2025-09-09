# CryptoSentiment Analysis

A sophisticated cryptocurrency sentiment analysis and price prediction system that combines real-time market data with social media sentiment to generate trading insights.

## 🎯 Purpose

This project implements a quantitative finance approach to cryptocurrency analysis by:
- Fetching real-time cryptocurrency market data from CoinMarketCap API
- Collecting social media sentiment from Reddit and Twitter
- Performing advanced sentiment analysis using NLTK's VADER analyzer
- Building machine learning models to predict price movements
- Combining technical and fundamental analysis with sentiment-driven insights

## 🏗️ Project Structure

```
CryptoSentiment/
├── crypto_sentiment_analyzer.py  # Main analysis module (refactored from notebook)
├── login_state.py               # Twitter authentication handler
├── test.py                      # Twitter sentiment scraper
├── config.py                    # Configuration management
├── Crypto_PRED.ipynb           # Jupyter notebook for interactive analysis
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Features

- **Multi-source Data Collection**: CoinMarketCap API, Reddit Pushshift API, Twitter scraping
- **Advanced Sentiment Analysis**: VADER sentiment intensity analysis
- **Machine Learning Predictions**: Linear regression models with sentiment features
- **Secure Configuration**: Environment variable management for API keys
- **Error Handling**: Robust error handling and logging
- **Type Hints**: Full type annotations for better code maintainability

## 📋 Prerequisites

- Python 3.8+
- CoinMarketCap API key
- Twitter account (for social media scraping)
- Internet connection for API calls

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/DanielEleojo/CryptoSentiment.git
cd CryptoSentiment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export COINMARKETCAP_API_KEY="your_api_key_here"
```

4. For Twitter scraping, run the login setup:
```bash
python login_state.py
```

## 📊 Usage

### Basic Analysis
```python
from crypto_sentiment_analyzer import CryptoMarketDataFetcher, SentimentAnalyzer

# Initialize components
market_fetcher = CryptoMarketDataFetcher()
sentiment_analyzer = SentimentAnalyzer()

# Fetch crypto data
crypto_data = market_fetcher.get_crypto_data(limit=10)

# Analyze sentiment
reddit_posts = RedditSentimentFetcher.get_reddit_posts('cryptocurrency')
sentiment_scores = sentiment_analyzer.batch_sentiment_analysis(reddit_posts['text'])
```

### Interactive Analysis
Use the Jupyter notebook for exploratory analysis:
```bash
jupyter notebook Crypto_PRED.ipynb
```

### Twitter Sentiment Scraping
```bash
python test.py
```

## 🔧 Configuration

The system uses environment variables for secure configuration:

- `COINMARKETCAP_API_KEY`: Your CoinMarketCap API key
- `REDDIT_CLIENT_ID`: Reddit API client ID (optional)
- `REDDIT_CLIENT_SECRET`: Reddit API client secret (optional)

## 📈 Model Performance

The system includes model evaluation metrics:
- R² Score for prediction accuracy
- Mean Absolute Error for prediction precision
- Sentiment correlation analysis

## 🔒 Security Notes

- API keys are managed through environment variables
- No hardcoded credentials in source code
- Twitter authentication state is saved locally and should not be shared

## 🤝 Contributing

This project follows quantitative finance best practices:
- PEP 8 code formatting
- Comprehensive type hints
- Detailed docstrings
- Error handling and logging
- Modular architecture

## 📄 License

Educational and research purposes. Please respect API terms of service.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Cryptocurrency trading involves significant risk. Past performance does not guarantee future results.
