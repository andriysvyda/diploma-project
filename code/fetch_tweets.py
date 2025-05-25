import tweepy
import pandas as pd

# Налаштування доступу
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAOng1wEAAAAAHdhQg6PwQBKQqv7SWxEU%2BFj5h38%3DyHw4sg44um04kFwJohy1mrKeK9aNcyGebsF1nvIHdackm0h6t1"  # Заміни на свій!
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Запит для збору твітів (українською, без ретвітів)
query = "#Україна OR #Київ OR #політика OR #економіка lang:uk -is:retweet -is:reply"
tweets = client.search_recent_tweets(
    query=query,
    max_results=100,  # Максимум для безкоштовного доступу
    tweet_fields=["created_at", "public_metrics"]
)

# Збереження у CSV
data = []
for tweet in tweets.data:
    data.append({
        "text": tweet.text,
        "date": tweet.created_at,
        "likes": tweet.public_metrics["like_count"]
    })

df = pd.DataFrame(data)
df.to_csv("tweets_ukraine.csv", index=False)
print(f"Зібрано {len(df)} твітів. Файл: tweets_ukraine.csv")