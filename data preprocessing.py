# data preprocessing script for cleaning text data

import pandas as pd
import re
from datasets import Dataset, ClassLabel
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


# Get unique values in the SentimentLabel column
labels = sorted(df['SentimentLabel'].unique())

# Convert pandas DataFrames to HuggingFace Dataset objects
train_dataset = Dataset.from_pandas(df)
test_dataset = Dataset.from_pandas(dt)

# Create an ordered list of sentiment names that match the numeric labels (0-4)
sentiment_names = [name for name, value in sorted(sentiment_map.items(), key=lambda x: x[1])]

# Create ClassLabel feature with proper mapping
class_label_feature = ClassLabel(num_classes=len(labels), names=sentiment_names)

# Cast the column to ClassLabel type
train_dataset = train_dataset.cast_column("SentimentLabel", class_label_feature)
test_dataset = test_dataset.cast_column("SentimentLabel", class_label_feature)
# Save cleaned dataset
#df.to_csv('data/Corona_NLP_train_cleaned.csv', index=False)
#dt.to_csv('data/Corona_NLP_test_cleaned.csv', index=False)

# Save HuggingFace Dataset objects (preserves ClassLabel information)
train_dataset.save_to_disk('data/train_dataset')
test_dataset.save_to_disk('data/test_dataset')