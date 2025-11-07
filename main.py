# --------------------------------------------------------------
# main.py
# --------------------------------------------------------------
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
from typing import List, Dict, Any

# ---------- 로깅 ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- FastAPI ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Redis ----------
try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True
    )
    redis_client.ping()
    logger.info("Successfully connected to Redis")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# ---------- python‑ta ----------
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import MoneyFlowIndexIndicator

# --------------------------------------------------------------
# 1. 티커 리스트 (KR / US)
# --------------------------------------------------------------
KOREAN_TO_ENGLISH = {
    "삼성전자": "Samsung Electronics",
    "현대차": "Hyundai Motor",
    "LG화학": "LG Chem",
    "SK하이닉스": "SK Hynix",
    "네이버": "Naver",
    "카카오": "Kakao",
    "현대모비스": "Hyundai Mobis",
    "기아": "Kia",
    "LG전자": "LG Electronics",
    "삼성SDI": "Samsung SDI",
}

@app.get("/tickers")
async def get_tickers(market: str = "KR"):
    try:
        cache_key = f"tickers:{market}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {cache_key}")
                return json.loads(cached)

        if market == "KR":
            # pykrx 로 한국 주식 티커 가져오기 (간단히 100개)
            tickers = stock.get_market_ticker_list()
            data = [{"ticker": t, "name": stock.get_market_ticker_name(t)} for t in tickers[:200]]
        else:
            # 미국은 yfinance 로 인기 종목만 (예시)
            data = [
                {"ticker": "AAPL", "name": "Apple"},
                {"ticker": "MSFT", "name": "Microsoft"},
                {"ticker": "GOOGL", "name": "Alphabet"},
                {"ticker": "AMZN", "name": "Amazon"},
                {"ticker": "TSLA", "name": "Tesla"},
            ]

        if redis_client:
            redis_client.setex(cache_key, 1800, json.dumps({"tickers": data}))
        return {"tickers": data}
    except Exception as e:
        logger.error(f"tickers error: {e}")
        return {"tickers": []}


# --------------------------------------------------------------
# 2. 주가·지표 분석 (/analyze)
# --------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    python‑ta 로 모든 지표를 한 번에 계산하고,
    현재(마지막 행) 값을 반환한다.
    """
    # 필수 컬럼만 남기기
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 전체 TA 피처 추가
    df = add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )

    # 최신값
    latest = df.iloc[-1]

    # ---- 개별 지표 (앱에서 사용하는 이름) ----
    indicators = {
        # 기본값
        "RSI": latest.get("momentum_rsi"),
        "MACD": latest.get("trend_macd"),
        "MACD_prev": df["trend_macd"].iloc[-2] if len(df) > 1 else None,
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

    # NaN → None
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
                logger.info(f"Cache hit for {cache_key}")
                return json.loads(cached)

        # ---------- 데이터 다운로드 ----------
        if market == "KR":
            end = datetime.today().strftime("%Y%m%d")
            start = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_date(start, end, ticker)
            df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            data = yf.download(ticker, period=period, interval="1d")
            df = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        if df.empty:
            raise ValueError("No price data")

        df = df.dropna().reset_index(drop=True)

        # ---------- 지표 ----------
        indicators = calculate_indicators(df)

        # ---------- 날짜 / 종가 ----------
        dates = df.index.strftime("%Y-%m-%d").tolist()
        closes = df["Close"].round(2).tolist()
        volumes = df["Volume"].tolist()

        result = {
            "dates": dates,
            "closes": closes,
            "volumes": volumes,
            "indicators": indicators,
        }

        if redis_client:
            redis_client.setex(cache_key, 1800, json.dumps(result))

        return result

    except Exception as e:
        logger.error(f"analyze error ({ticker}): {e}")
        return {"error": str(e)}


# --------------------------------------------------------------
# 3. 뉴스 (네이버 검색 API 예시)
# --------------------------------------------------------------
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_SECRET")

@app.get("/news")
async def get_news(ticker: str, market: str = "KR", start: int = 1, display: int = 10):
    try:
        query = ticker if market == "KR" else f"{ticker} stock"
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        }
        params = {"query": query, "display": display, "start": start, "sort": "date"}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        news_list = []
        for it in items:
            title = it.get("title", "").replace("<b>", "").replace("</b>", "")
            link = it.get("link", "")
            pub_date = it.get("pubDate", "")
            # 간단 감성 키워드 (예시)
            sentiment = "중립"
            if any(w in title.lower() for w in ["상승", "호재", "강세"]):
                sentiment = "긍정"
            elif any(w in title.lower() for w in ["하락", "악재", "약세"]):
                sentiment = "부정"
            news_list.append({
                "title": title,
                "url": link,
                "date": pub_date,
                "sentiment": sentiment,
                "color": {"긍정": "green", "부정": "red", "중립": "yellow"}.get(sentiment, "yellow")
            })

        return {"news": news_list, "total": data.get("total", 0)}
    except Exception as e:
        logger.error(f"news error: {e}")
        return {"news": [], "total": 0}


# --------------------------------------------------------------
# 4. 변동률 비교 (/compare)
# --------------------------------------------------------------
@app.get("/compare")
async def compare(ticker: str, market: str = "US"):
    try:
        tickers = [ticker, "^KS11", "^IXIC"]
        changes = {}

        for t in tickers:
            if t == ticker and market == "KR":
                end = datetime.today().strftime("%Y%m%d")
                start = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
                df = stock.get_market_ohlcv_by_date(start, end, t)
                df = df.rename(columns={"종가": "Close"})
            else:
                df = yf.download(t, period="3mo", interval="1d")

            if df.empty:
                changes[t] = 0.0
                continue
            first = df["Close"].iloc[0]
            last = df["Close"].iloc[-1]
            change = ((last - first) / first) * 100
            changes[t] = round(float(change), 2)

        return {
            ticker: changes[ticker],
            "KOSPI": changes["^KS11"],
            "NASDAQ": changes["^IXIC"]
        }
    except Exception as e:
        logger.error(f"compare error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# --------------------------------------------------------------
# 5. 서버 상태 체크 (옵션)
# --------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}