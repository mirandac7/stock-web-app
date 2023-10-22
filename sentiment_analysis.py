import pandas as pd
import numpy as np
from datetime import date
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity #polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0:
        sentiment_label = 'Positive'
    elif sentiment_polarity < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    return result

vader = SentimentIntensityAnalyzer()
def add_sentiment(parsed_news, model='TextBlob'):
    if model == "TextBlob":
        parsed_news['sentiment_results'] = parsed_news['Title'].apply(get_sentiment)
        parsed_news = parsed_news.join(pd.json_normalize(parsed_news['sentiment_results']))
    if model == "NLTK":
        parsed_news['scores_vader'] = parsed_news['Title'].apply(vader.polarity_scores).tolist()
        parsed_news = parsed_news.join(pd.json_normalize(parsed_news['scores_vader']))
        # parsed_news.renamed(columns={'ne'})
    return parsed_news

