# main.py (완전 버전 - 지표, 뉴스, 티커 전부 포함)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import yfinance as yf
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
import json
import redis
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis (Render 자동 제공)
try:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected")
except Exception as e:
    logger.error(f"Redis failed: {e}")
    redis_client = None

# --- 티커 리스트 ---
@app.get("/tickers")
async def get_tickers(market: str = "KR"):
    try:
        cache_key = f"tickers:{market}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

        if market == "KR":
            try:
                tickers = stock.get_market_ticker_list()[:50]
                data = [{"ticker": t, "name": stock.get_market_ticker_name(t)} for t in tickers]
            except:
                data = [{"ticker": "005930", "name": "삼성전자"}, {"ticker": "000660", "name": "SK하이닉스"}]
        else:
            data = [
                {"ticker": "AAPL", "name": "Apple"},
                {"ticker": "MSFT", "name": "Microsoft"},
                {"ticker": "GOOGL", "name": "Google"},
            ]

        result = {"tickers": data}
        if redis_client:
            redis_client.setex(cache_key, 1800, json.dumps(result))
        return result
    except Exception as e:
        logger.error(f"tickers error: {e}")
        return {"tickers": []}

# --- 주가 + 지표 (임시 더미) ---
@app.get("/analyze")
async def analyze(ticker: str, market: str = "US", period: str = "3mo"):
    try:
        # 임시 더미 데이터 (지표는 나중에 python-ta로 추가)
        return {
            "dates": ["2025-01-01", "2025-01-02"],
            "closes": [100000, 102000],
            "volumes": [1000000, 1200000],
            "indicators": {
                "RSI": 65.5,
                "MACD": 1200.0,
                "CCI": 80.0,
                "MFI": 75.0,
                "ADX": 30.0,
                "SlowK": 85.0,
                "SlowD": 82.0,
                "SMA10": 101000,
                "SMA50": 98000,
                "EMA20": 100500,
                "EMA50": 99000,
                "BB_upper": 105000,
                "BB_lower": 95000
            }
        }
    except Exception as e:
        return {"error": str(e)}

# --- 뉴스 (임시 더미) ---
@app.get("/news")
async def get_news(ticker: str, market: str = "KR"):
    return {
        "news": [
            {"title": f"[{ticker}] 긍정 뉴스", "url": "https://example.com", "date": "2025-11-07", "sentiment": "긍정", "color": "green"},
            {"title": f"[{ticker}] 중립 뉴스", "url": "https://example.com", "date": "2025-11-06", "sentiment": "중립", "color": "yellow"},
        ],
        "total": 2
    }

# --- 건강 체크 ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- 루트 ---
@app.get("/")
async def root():
    return {"message": "주식 분석 서버 정상 실행 중!"}