import nltk
import os

def setup_nltk():
    """
    Download necessary NLTK data for sentiment analysis
    """
    print("Setting up NLTK for sentiment analysis...")
    try:
        nltk.download('vader_lexicon')
        print("Successfully downloaded VADER lexicon for sentiment analysis")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

if __name__ == "__main__":
    setup_nltk()
    print("\nSetup complete! You can now run the app with: streamlit run app.py")