import tweepy
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "apiKey"
API_SECRET = "apisecret"
ACCESS_TOKEN = "accessToken"
ACCESS_TOKEN_SECRET = "accessTokenSecret"
BEARER_TOKEN = "BEAER_TOKEN"

client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)

def fetch_tweets(query="#Україна", count=25):
    try:
        tweets = client.search_recent_tweets(query=query, max_results=count, tweet_fields=['created_at', 'public_metrics'])
        if not tweets.data:
            logger.info("Твіти не знайдені.")
            return
        data = []
        for tweet in tweets.data:
            data.append({
                'text': tweet.text,
                'date': tweet.created_at,
                'likes': tweet.public_metrics['like_count']
            })
        df = pd.DataFrame(data)
        df.to_csv("tweets_ukraine.csv", index=False)
        logger.info(f"Збережено {len(df)} твітів у tweets_ukraine.csv")
    except Exception as e:
        logger.error(f"Помилка при зборі твітів: {str(e)}")

if __name__ == "__main__":
    fetch_tweets()