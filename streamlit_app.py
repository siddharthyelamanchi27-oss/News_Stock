import streamlit as st
import requests
import nltk
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from groq import Groq
import pandas as pd

# ------------------ SETUP ------------------

nltk.data.path.append("/Users/syelamanchi/nltk_data")
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


TOP_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "XOM"]

sia = SentimentIntensityAnalyzer()

# ------------------ FUNCTIONS ------------------

def get_top_articles(stock):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{stock} stock",
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

def analyze_sentiment(text):
    return sia.polarity_scores(text)["compound"]

def extract_article_text(article):
    return article.get("description") or article.get("content")

def analyze_articles(articles):
    scores = []
    for article in articles:
        text = extract_article_text(article)
        if text:
            scores.append(analyze_sentiment(text))
    return sum(scores) / len(scores) if scores else 0

def get_stock_data(stock):
    ticker = yf.Ticker(stock)
    hist = ticker.history(period="1mo")
    price = hist["Close"][-1]
    trend = hist["Close"].pct_change().mean()
    volatility = hist["Close"].pct_change().std()
    return price, trend, volatility, hist

def investment_score(sentiment, trend, volatility):
    raw = sentiment * 0.4 + trend * 0.3 - volatility * 0.2
    raw = max(-1, min(1, raw))
    return (raw + 1) * 50

def ai_explain(stock, price, sentiment, trend, volatility, score, articles):
    titles = [a["title"] for a in articles if a.get("title")]

    prompt = f"""
You are a financial analyst explaining stock data to a student.

Stock: {stock}
Price: ${price:.2f}
Sentiment: {sentiment:.3f}
Trend: {trend:.4f}
Volatility: {volatility:.4f}
Investment Score: {score:.1f}/100

Recent news headlines:
{titles}

Explain:
1. What the score means
2. Why sentiment matters
3. Whether this looks risky or promising
4. One warning
5. One opportunity

Keep it clear and readable.
"""

    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=500,
        stream=True
    )

    return stream

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

st.title("📈 AI-Powered Stock News Analyzer")

stock = st.text_input("Enter stock ticker (e.g. AAPL, TSLA)").upper()

if st.button("Analyze") and stock:
    with st.spinner("Analyzing stock and news..."):
        articles = get_top_articles(stock)
        sentiment = analyze_articles(articles)
        price, trend, volatility, hist = get_stock_data(stock)
        score = investment_score(sentiment, trend, volatility)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Price", f"${price:.2f}")
        st.metric("Investment Score", f"{score:.1f} / 100")
        st.line_chart(hist["Close"])

    with col2:
        st.subheader("🧠 AI Explanation")
        placeholder = st.empty()

        explanation = ""
        for chunk in ai_explain(stock, price, sentiment, trend, volatility, score, articles):
            delta = chunk.choices[0].delta.content
            if delta:
                explanation += delta
                placeholder.markdown(explanation)

    st.markdown("### 📰 Top Articles")
    for a in articles:
        if a.get("url"):
            st.markdown(f"- [{a.get('title')}]({a.get('url')})")

# ------------------ STATIC TOP STOCK CHARTS ------------------

st.markdown("---")
st.header("📊 Top 10 Stocks (1 Month Performance)")

cols = st.columns(2)

for i, ticker in enumerate(TOP_STOCKS):
    hist = yf.Ticker(ticker).history(period="1mo")
    with cols[i % 2]:
        st.subheader(ticker)
        st.line_chart(hist["Close"])
