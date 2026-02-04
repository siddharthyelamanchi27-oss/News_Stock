import streamlit as st
import requests
import yfinance as yf
from groq import Groq

# ================== CONFIG ==================
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

# ================== SECRETS / API KEYS ==================
# Streamlit Cloud secrets
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    import os
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not NEWS_API_KEY or not GROQ_API_KEY:
    st.error("Missing API keys. Set NEWS_API_KEY and GROQ_API_KEY.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ================== SETTINGS ==================
TOP_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "XOM"]
POSITIVE_WORDS = {"growth", "gain", "strong", "beat", "bullish", "profit", "surge", "record", "improve", "positive"}
NEGATIVE_WORDS = {"loss", "down", "weak", "miss", "bearish", "drop", "fall", "decline", "risk", "negative"}

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
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        st.warning(f"Could not fetch news: {e}")
        return []

def analyze_sentiment(text):
    text = text.lower()
    score = sum(1 for w in POSITIVE_WORDS if w in text) - sum(1 for w in NEGATIVE_WORDS if w in text)
    return max(-1, min(1, score / 5))

def analyze_articles(articles):
    scores = [analyze_sentiment(a.get("description") or a.get("content") or "") for a in articles]
    return sum(scores) / len(scores) if scores else 0

def get_stock_data(stock):
    hist = yf.Ticker(stock).history(period="1mo")
    return hist["Close"].iloc[-1], hist["Close"].pct_change().mean(), hist["Close"].pct_change().std(), hist

def investment_score(sentiment, trend, volatility):
    raw = sentiment * 0.4 + trend * 0.3 - volatility * 0.2
    return (max(-1, min(1, raw)) + 1) * 50

def generate_explanation(stock, price, sentiment, trend, volatility, score, articles):
    titles = [a["title"] for a in articles if a.get("title")][:5]
    prompt = f"""
You are a financial analyst explaining stock data to a student.

Stock: {stock}
Price: ${price:.2f}
Sentiment: {sentiment:.3f}
Trend: {trend:.4f}
Volatility: {volatility:.4f}
Investment Score: {score:.1f}/100

Recent headlines:
{titles}

Explain clearly:
- What the score means
- Risk level
- Whether outlook is positive or negative
"""
    try:
        stream = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_completion_tokens=500
        )
        explanation = ""
        for chunk in stream:
            delta = None
            if hasattr(chunk.choices[0].delta, "content"):
                delta = chunk.choices[0].delta.content
            elif isinstance(chunk.choices[0].delta, dict):
                delta = chunk.choices[0].delta.get("content")
            if delta:
                explanation += delta
        return explanation
    except Exception as e:
        return f"Error generating explanation: {e}"

def chat_with_context(user_message):
    context = st.session_state.analysis_context
    prompt = f"""
You are a stock analysis chatbot.

You MUST base answers ONLY on the following analysis.
Do NOT guess or predict beyond this data.

ANALYSIS CONTEXT:
{context}

User question:
{user_message}

Answer clearly and cautiously.
"""
    try:
        return client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_completion_tokens=400
        )
    except Exception as e:
        return [{"choices":[{"delta":{"content":f"Error: {e}"}}]}]

# ================== SESSION STATE ==================
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_stock" not in st.session_state:
    st.session_state.current_stock = ""

# ================== UI ==================
st.title("📈 AI-Powered Stock News Analyzer")

# Stock selection
stock = st.selectbox("Select a stock", TOP_STOCKS + ["Other"])
if stock == "Other":
    stock = st.text_input("Enter stock ticker").upper()

if st.button("Analyze") and stock:
    if stock != st.session_state.current_stock:
        st.session_state.analysis_context = ""
        st.session_state.chat_history = []
        st.session_state.current_stock = stock

    with st.spinner("Analyzing..."):
        articles = get_top_articles(stock)
        sentiment = analyze_articles(articles)
        price, trend, volatility, hist = get_stock_data(stock)
        score = investment_score(sentiment, trend, volatility)
        explanation = generate_explanation(stock, price, sentiment, trend, volatility, score, articles)
        st.session_state.analysis_context = explanation

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Price", f"${price:.2f}")
        st.metric("Score", f"{score:.1f} / 100")
        st.line_chart(hist["Close"])
    with col2:
        st.subheader("🧠 AI Explanation")
        st.markdown(explanation)

# ================== CHATBOT ==================
st.markdown("---")
st.header("💬 Ask About This Stock")

# Show previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input always visible
user_input = st.chat_input("Ask about the stock (e.g. Will it go up?)")
if user_input and st.session_state.analysis_context:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant streaming response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        for chunk in chat_with_context(user_input):
            delta = None
            if hasattr(chunk.choices[0].delta, "content"):
                delta = chunk.choices[0].delta.content
            elif isinstance(chunk.choices[0].delta, dict):
                delta = chunk.choices[0].delta.get("content")
            if delta:
                reply += delta
                placeholder.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ================== TOP STOCKS ==================
st.markdown("---")
st.header("📊 Top 10 Stocks — Last Month")
cols = st.columns(2)
for i, ticker in enumerate(TOP_STOCKS):
    hist = yf.Ticker(ticker).history(period="1mo")
    with cols[i % 2]:
        st.subheader(ticker)
        st.line_chart(hist["Close"])

