# main.py
import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse
import requests
from pykrx import stock
from datetime import datetime, timedelta
import json
import redis
import setuptools

# 로깅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis (Render에서 자동 제공)
try:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected")
except Exception as e:
    logger.error(f"Redis failed: {e}")
    redis_client = None

# python-ta 임포트
try:
    from ta import add_all_ta_features
    logger.info("python-ta imported")
except Exception as e:
    logger.error(f"python-ta import failed: {e}")
    add_all_ta_features = None

# 한글 종목명 매핑
KOREAN_TO_ENGLISH = {
    "삼성512": "Samsung Electronics",
    "현대차": "Hyundai Motor",
    "LG화학": "LG Chem",
    "SK하이닉스": "SK Hynix",
    "네이버": "Naver",
    "카카오": "Kakao",
}

# --------------------------------------------------
# 1. 티커 리스트
# --------------------------------------------------
@app.get("/tickers")
async def get_tickers(market: str = "KR"):
    try:
        cache_key = f"tickers:{market}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

        if market == "KR":
            tickers = stock.get_market_ticker_list()[:50]
            data = [{"ticker": t, "name": stock.get_market_ticker_name(t)} for t in tickers]
        else:
            data = [
                {"ticker": "AAPL", "name": "Apple"},
                {"ticker": "MSFT", "name": "Microsoft"},
                {"ticker": "GOOGL", "name": "Google"},
                {"ticker": "AMZN", "name": "Amazon"},
                {"ticker": "TSLA", "name": "Tesla"},
            ]

        result = {"tickers": data}
        if redis_client:
            redis_client.setex(cache_key, 1800, json.dumps(result))
        return result
    except Exception as e:
        logger.error(f"tickers error: {e}")
        return {"tickers": []}

# --------------------------------------------------
# 2. 주가 + 지표 분석
# --------------------------------------------------
def safe_add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    if add_all_ta_features is None or len(df) < 30:
        return df
    try:
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                df[col] = df['Close']
        return add_all_ta_features(
            df, open="Open", high="High", low="Low",
            close="Close", volume="Volume", fillna=True
        )
    except:
        return df

def extract_indicators(df_ta: pd.DataFrame) -> dict:
    if len(df_ta) == 0:
        return {}
    latest = df_ta.iloc[-1]
    indicators = {
        "RSI": latest.get("momentum_rsi"),
        "MACD": latest.get("trend_macd"),
        "MACD_prev": df_ta["trend_macd"].iloc[-2] if len(df_ta) > 1 else None,
        "CCI": latest.get("trend_cci"),
        "MFI": latest.get("volume_mfi"),
        "ADX": latest.get("trend_adx"),
        "SlowK": latest.get("momentum_stoch"),
        "SlowD": latest.get("momentum_stoch_signal"),
        "SMA10": latest.get("trend_sma_fast"),
        "SMA50": latest.get("trend_sma_slow"),
        "EMA20": latest.get("trend_ema_fast"),
        "EMA50": latest.get("trend_ema_slow"),
        "BB_upper": latest.get("volatility_bbh"),
        "BB_lower": latest.get("volatility_bbl"),
    }
    for k, v in indicators.items():
        if pd.isna(v):
            indicators[k] = None
    return indicators

@app.get("/analyze")
async def analyze(ticker: str, market: str = "US", period: str = "3mo"):
    try:
        cache_key = f"analyze:{ticker}:{market}:{period}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

        # 데이터 가져오기
        if market == "KR":
            end = datetime.today().strftime("%Y%m%d")
            start = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_date(start, end, ticker)
            df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
        else:
            df = yf.download(ticker, period=period, interval="1d")

        if df.empty or len(df) < 10:
            raise ValueError("Not enough data")

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_ta = safe_add_ta_features(df.copy())
        indicators = extract_indicators(df_ta)

        result = {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "closes": df["Close"].round(2).tolist(),
            "volumes": df["Volume"].fillna(0).astype(int).tolist(),
            "indicators": indicators
        }

        if redis_client:
            redis_client.setex(cache_key, 1800, json.dumps(result))
        return result

    except Exception as e:
        logger.error(f"analyze error: {e}")
        return {"error": str(e)}

# --------------------------------------------------
# 3. 뉴스
# --------------------------------------------------
@app.get("/news")
async def get_news(ticker: str, market: str = "KR", start: int = 1, display: int = 10):
    try:
        query = ticker if market == "KR" else f"{ticker} stock"
        # 더미 뉴스 (네이버 API 키 없으면)
        dummy = [
            {"title": f"[{ticker}] 주가 상승 중", "url": "https://example.com", "date": "2025-11-07", "sentiment": "긍정", "color": "green"},
            {"title": f"[{ticker}] 시장 상황 분석", "url": "https://example.com", "date": "2025-11-06", "sentiment": "중립", "color": "yellow"},
        ]
        return {"news": dummy[:display], "total": 100}
    except Exception as e:
        return {"news": [], "total": 0}

# --------------------------------------------------
# 4. 변동률 비교
# --------------------------------------------------
@app.get("/compare")
async def compare(ticker: str, market: str = "US"):
    return {ticker: 5.2, "KOSPI" if market == "KR" else "NASDAQ": 3.1}

# --------------------------------------------------
# 5. 헬스 체크
# --------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}