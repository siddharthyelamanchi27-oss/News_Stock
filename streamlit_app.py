import streamlit as st
import requests
import yfinance as yf
import os
from groq import Groq

# ================== CONFIG ==================

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

TOP_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "XOM"]

# Simple keyword sentiment (NO NLTK)
POSITIVE_WORDS = {
    "growth", "gain", "up", "strong", "beat", "bullish",
    "profit", "surge", "record", "improve", "positive"
}

NEGATIVE_WORDS = {
    "loss", "down", "weak", "miss", "bearish",
    "drop", "fall", "decline", "risk", "negative"
}

# ================== FUNCTIONS ==================

def get_top_articles(stock):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{stock} stock",
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params).json()
    return response.get("articles", [])

def extract_article_text(article):
    return article.get("description") or article.get("content") or ""

def analyze_sentiment(text):
    text = text.lower()
    score = 0

    for w in POSITIVE_WORDS:
        if w in text:
            score += 1

    for w in NEGATIVE_WORDS:
        if w in text:
            score -= 1

    return max(-1, min(1, score / 5))

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
    titles = [a["title"] for a in articles if a.get("title")][:5]

    prompt = f"""
You are a financial analyst explaining stock data to a student.

Stock: {stock}
Price: ${price:.2f}
Sentiment Score: {sentiment:.3f}
Trend: {trend:.4f}
Volatility: {volatility:.4f}
Investment Score: {score:.1f}/100

Recent Headlines:
{titles}

Explain:
1. What the score means
2. Why sentiment matters
3. Risk level
4. One warning
5. One opportunity

Keep it clear and simple.
"""

    return client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=500,
        stream=True
    )

# ================== UI ==================

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
        st.metric("Current Price", f"${price:.2f}")
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

    st.markdown("### 📰 Relevant News Articles")
    shown = 0
    for a in articles:
        if stock.lower() in (a.get("title", "").lower()):
            st.markdown(f"- [{a.get('title')}]({a.get('url')})")
            shown += 1
        if shown == 5:
            break

# ================== TOP STOCKS ==================

st.markdown("---")
st.header("📊 Top 10 Stocks — Last Month")

cols = st.columns(2)

for i, ticker in enumerate(TOP_STOCKS):
    hist = yf.Ticker(ticker).history(period="1mo")
    with cols[i % 2]:
        st.subheader(ticker)
        st.line_chart(hist["Close"])

