# data preprocessing script for cleaning text data

import pandas as pd
import re

# Load dataset
df = pd.read_csv('data/Corona_NLP_train.csv', encoding='utf-8', engine='python')
dt = pd.read_csv('data/Corona_NLP_test.csv', encoding='utf-8', engine='python')

def clean_text(text):
    # Remove unwanted character Â and normalize whitespace
    text = re.sub(r'Â', '', str(text))
    #text = re.sub(r'\s+', ' ', text) need to review the dataset for inconsistent whitespace
    return text.strip()

# Apply cleaning to OriginalTweet
df['OriginalTweet'] = df['OriginalTweet'].apply(clean_text)
dt['OriginalTweet'] = dt['OriginalTweet'].apply(clean_text)

# Create mapping from sentiment strings to integers
sentiment_map = {
    "Extremely Negative": 0,
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3,
    "Extremely Positive": 4
}

# Add numerical labels column
df['SentimentLabel'] = df['Sentiment'].map(sentiment_map)
dt['SentimentLabel'] = dt['Sentiment'].map(sentiment_map)

# Save cleaned dataset
df.to_csv('data/Corona_NLP_train_cleaned.csv', index=False)
dt.to_csv('data/Corona_NLP_test_cleaned.csv', index=False)