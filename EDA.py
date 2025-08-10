import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

train = pd.read_csv("data/Corona_NLP_train.csv")
test = pd.read_csv("data/Corona_NLP_test.csv")
dates = np.unique(train['TweetAt'])
# Exploratory Data Analysis (EDA) for COVID-19 Tweets Dataset
#plot the number of tweets per day in the training set in a  calendar heatmap
def plot_calendar_heatmap(data, dates):
    temp = pd.DataFrame()
    temp['day'] = pd.to_datetime(data['TweetAt'],dayfirst=True)
    temp['year'] = data['day'].dt.isocalendar().year
    temp['week'] = data['day'].dt.isocalendar().week
    temp['weekday'] = data['day'].dt.dayofweek

    # Group by year, week, and weekday and count the tweets
    daily_counts = temp.groupby(['year', 'week', 'weekday']).size()

    # Create a pivot table for the heatmap
    heatmap_data = daily_counts.unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_data.values, cmap='Blues', aspect='auto')
    plt.colorbar(label='Number of Tweets')

    # Adjust ticks and labels for the new structure
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    # Generate labels for the y-axis showing the date range of each week
    y_labels = []
    for year, week in heatmap_data.index:
        start_of_week = datetime.fromisocalendar(year, week, 1)
        end_of_week = datetime.fromisocalendar(year, week, 7)
        y_labels.append(f"{start_of_week.strftime('%d/%m')}-{end_of_week.strftime('%d/%m')}")

    plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=y_labels)

    plt.title('Number of Tweets per Day (Calendar Heatmap)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Week of the Year')
    plt.tight_layout()
    plt.show()

plot_calendar_heatmap(train, dates)
plot_calendar_heatmap(test, dates)

def plot_tweet_length_distribution(data):
    temp=pd.DataFrame()
    # Calculate the length of each tweet
    temp['length'] = data['OriginalTweet'].apply(lambda x: len(str(x).split()))
    # Plot the distribution of tweet lengths
    plt.figure(figsize=(10, 6))
    plt.hist(temp['length'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Length of Tweet (in words)')
    plt.ylabel('Number of Tweets')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
plot_tweet_length_distribution(train)
plot_tweet_length_distribution(test)

# Plot the distribution of sentiment labels
def plot_sentiment_distribution(data):
    sentiment_counts = data['Sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Distribution of Sentiment Labels')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.75)
    plt.show()
plot_sentiment_distribution(train)
plot_sentiment_distribution(test)

# Plot the distribution of sentiment labels over time
import matplotlib.dates as mdates
def plot_sentiment_over_time(data):
    temp = pd.DataFrame()
    temp['TweetAt'] = pd.to_datetime(data['TweetAt'], dayfirst=True)
    temp['Sentiment'] = data['Sentiment']

    # Group by date and sentiment, then resample to get daily counts
    sentiment_over_time = temp.groupby([temp['TweetAt'].dt.date, 'Sentiment']).size().unstack(fill_value=0)
    sentiment_over_time.index = pd.to_datetime(sentiment_over_time.index)

    # Plot the resampled data
    fig, ax = plt.subplots(figsize=(12, 6))
    sentiment_over_time.plot(kind='line', marker='o', ax=ax)

    # Set major ticks to be every 3 days and disable minor tick labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    plt.title('Sentiment Distribution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_sentiment_over_time(train)
plot_sentiment_over_time(test)

# plot the distribution of tweet lengths grouped by sentiment
def plot_tweet_length_by_sentiment(data):
    temp = pd.DataFrame()
    # Calculate the length of each tweet
    temp['length'] = data['OriginalTweet'].apply(lambda x: len(str(x).split()))
    temp['Sentiment'] = data['Sentiment']

    sentiments = sorted(temp['Sentiment'].unique())
    num_sentiments = len(sentiments)

    # Determine grid size for a more square layout
    ncols = 2
    nrows = int(np.ceil(num_sentiments / ncols))

    # Create a figure with a subplot for each sentiment
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12), sharex=False, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Plot a histogram for each sentiment on its own subplot
    for i, sentiment in enumerate(sentiments):
        ax = axes[i]
        subset = temp[temp['Sentiment'] == sentiment]
        ax.hist(subset['length'], bins=50, alpha=0.75)
        ax.set_title(f'Sentiment: {sentiment}')
        ax.grid(axis='y', alpha=0.75)

    # Hide any unused subplots
    for i in range(num_sentiments, nrows * ncols):
        fig.delaxes(axes[i])

    fig.suptitle('Distribution of Tweet Lengths by Sentiment', fontsize=16)
    # Use common labels for the shared axes
    fig.supxlabel('Length of Tweet (in words)')
    fig.supylabel('Number of Tweets')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
plot_tweet_length_by_sentiment(train)
plot_tweet_length_by_sentiment(test)

from wordcloud import WordCloud, STOPWORDS

def plot_word_cloud(data, title):
    # Create a set of stopwords
    stopwords = set(STOPWORDS)
    # Add custom words that are not meaningful for the analysis
    stopwords.update(["https", "co", "amp", "t","Â"])

    # Join all tweets into a single string
    text = " ".join(tweet for tweet in data.OriginalTweet.astype(str))

    # Create and generate a word cloud image, excluding stopwords
    wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color='white').generate(text)

    # Display the generated image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

# Plot word cloud for the training and test data
plot_word_cloud(train, 'Word Cloud for Training Data')
plot_word_cloud(test, 'Word Cloud for Test Data')