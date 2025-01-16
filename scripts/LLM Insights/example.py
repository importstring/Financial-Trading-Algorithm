import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def get_response(query):
    # Chatgpt-4o response when asked about their take on microsoft stock
    return r"""
Microsoft (NASDAQ: MSFT) presents a compelling investment opportunity, but investors should carefully consider several factors before deciding to buy or wait.

## Current Performance

As of January 7, 2025, Microsoft's stock price stands at $422.37, down 1.28% for the day. Despite this recent dip, the company has shown strong performance over the past year, with the stock up approximately 15% in 2024[9].

## Analyst Outlook

The overall sentiment among analysts remains bullish:

- The average analyst price target is approximately $509.90, indicating potential upside[2].
- Some analysts have set even higher targets, with Truist Financial projecting a price of $600[2].
- The stock currently has a consensus rating of "Moderate Buy"[2][7].

## Growth Drivers

Several factors contribute to the positive outlook for Microsoft:

1. **AI Investments**: Microsoft's $14 billion investment in OpenAI is expected to drive significant growth, with AI-related revenue projected to exceed $10 billion in Q2 fiscal 2025[9].

2. **Cloud Computing**: Azure, Microsoft's cloud platform, continues to show strong growth, with revenue increasing 33% year-over-year in Q1 fiscal 2025[9].

3. **Diversified Portfolio**: Microsoft's range of products and services, including Office, Windows, and gaming, provides multiple revenue streams[2].

## Potential Risks

1. **Market Competition**: The tech sector remains highly competitive, with companies like Amazon and Google posing significant challenges[2].

2. **Regulatory Scrutiny**: As a major tech player, Microsoft faces ongoing regulatory risks[2].

3. **Economic Factors**: Broader economic conditions could impact Microsoft's growth and profitability[2].

## Conclusion

While Microsoft shows strong potential for growth, particularly in AI and cloud computing, the decision to buy or wait depends on individual investment goals and risk tolerance. The current stock price is below analyst targets, suggesting potential upside. However, given recent volatility and broader market conditions, cautious investors might consider waiting for a more significant dip before entering or adding to a position.

For those with a long-term investment horizon, Microsoft's strong fundamentals and strategic positioning in key growth areas make it an attractive option, even at current levels. As always, it's advisable to conduct thorough research and consider consulting with a financial advisor before making investment decisions.

Citations:
[1] https://www.marketbeat.com/instant-alerts/microsoft-nasdaqmsft-shares-down-04-whats-next-2025-01-07/
[2] https://www.markets.com/analysis/hot-stock-right-now-analyst-expects-microsoft-stock-msft-to-surge
[3] https://watcher.guru/news/microsoft-stock-why-analysts-say-msft-could-hit-550-in-2025
[4] https://www.investors.com/research/microsoft-stock-msft-buy-january/
[5] https://stockhouse.com/news/the-market-online-news/2025/01/06/blackberry-and-microsoft-team-up-to-capture-vehicle-software-market-share
[6] https://www.nasdaq.com/articles/should-you-buy-microsoft-stock-right-now-2025
[7] https://www.marketbeat.com/instant-alerts/microsoft-nasdaqmsft-rating-increased-to-buy-at-stocknewscom-2025-01-06/
[8] https://247wallst.com/investing/2025/01/07/microsoft-msft-price-prediction-and-forecast/
[9] https://www.insidermonkey.com/blog/is-microsoft-corporation-nasdaqmsft-among-israel-englanders-top-stock-picks-heading-into-2025-1418915/
"""


# Download the VADER lexicon if not already present
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(statement):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Analyze the sentiment of the statement
    sentiment_scores = sia.polarity_scores(statement)
    
    # Extract the compound score
    compound_score = sentiment_scores['compound']
    
    # Determine the sentiment based on the compound score
    if compound_score >= 0.05:
        sentiment = "Bullish"
    elif compound_score <= -0.05:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    
    # Calculate a normalized score between -1 (extremely bearish) and 1 (extremely bullish)
    normalized_score = compound_score * 2
    
    return sentiment, normalized_score

def decode_and_evaluate(query):
    # Get the response from the hypothetical function
    statement = get_response(query)
    
    # Analyze the sentiment of the statement
    sentiment, score = analyze_sentiment(statement)
    
    print(f"Statement: {statement}")
    print(f"Sentiment: {sentiment}")
    print(f"Score: {score:.2f} (-1 extremely bearish, 1 extremely bullish)")

# Example usage
query = "What's your opinion on the current market conditions?"
decode_and_evaluate(query)
