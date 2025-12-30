import pandas as pd
import numpy as np
from datetime import datetime
import os
import nltk

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except (LookupError, ImportError):
    try:
        nltk.download('vader_lexicon')
    except Exception as e:
        print(f"Could not download vader_lexicon: {e}")

# Import TextBlob for sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Import VADER as fallback
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Import GoogleNews for dynamic news fetching
try:
    from GoogleNews import GoogleNews
    GOOLGENEWS_AVAILABLE = True
except ImportError:
    GOOLGENEWS_AVAILABLE = False

def analyze_news_sentiment():
    """
    Fetch the latest 10 TCS news headlines using GoogleNews and analyze sentiment.
    Returns:
    --------
    pandas.DataFrame
        DataFrame with news headlines and sentiment scores
    """
    if not GOOLGENEWS_AVAILABLE:
        return pd.DataFrame()
    try:
        googlenews = GoogleNews(lang='en')
        googlenews.clear()
        googlenews.search('TCS Tata Consultancy Services')
        news_items = googlenews.result()[:10]
        if not news_items:
            return pd.DataFrame()
        headlines = []
        dates = []
        for item in news_items:
            headlines.append(item.get('title', ''))
            # Try to parse date, fallback to today if not available
            date_str = item.get('date', '')
            try:
                if date_str:
                    date = pd.to_datetime(date_str, errors='coerce')
                    if pd.isnull(date):
                        date = datetime.now()
                else:
                    date = datetime.now()
            except Exception:
                date = datetime.now()
            dates.append(date)
        news_df = pd.DataFrame({'Date': dates, 'Headline': headlines})
        # Sentiment analysis
        sentiments = []
        for headline in news_df['Headline']:
            if TEXTBLOB_AVAILABLE:
                sentiment = TextBlob(str(headline)).sentiment.polarity
            elif VADER_AVAILABLE:
                analyzer = SentimentIntensityAnalyzer()
                sentiment = analyzer.polarity_scores(str(headline))['compound']
            else:
                sentiment = 0.0
            sentiments.append(sentiment)
        news_df['Sentiment'] = sentiments
        news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce').fillna(datetime.now())
        return news_df
    except Exception as e:
        print(f"Error fetching or analyzing news: {str(e)}")
        return pd.DataFrame()

def get_sentiment_score(text):
    """
    Calculate sentiment score for a given text
    
    Parameters:
    -----------
    text : str
        Text to analyze
        
    Returns:
    --------
    float
        Sentiment score between -1 (negative) and 1 (positive)
    """
    if not isinstance(text, str):
        return 0.0
    
    # Try TextBlob first if available
    if TEXTBLOB_AVAILABLE:
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            pass
    
    # Fall back to VADER if available
    if VADER_AVAILABLE:
        try:
            sid = SentimentIntensityAnalyzer()
            sentiment_dict = sid.polarity_scores(text)
            return sentiment_dict['compound']  # Compound score between -1 and 1
        except Exception:
            pass
    
    # If all else fails, return neutral sentiment
    return 0.0