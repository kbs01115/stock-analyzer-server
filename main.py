# 파일명: main.py
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
import logging
import os

# 로깅 설정
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

# Redis 클라이언트 설정
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

# python-ta 임포트 (TA-Lib 대체)
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import MoneyFlowIndexIndicator
from ta.others import UltimateOscillator  # CCI 대체용

# 한글 종목명 매핑 (기존)
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
        # Redis 캐시 확인
        if redis_client:
            cache_key = f"tickers:{market}"
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    logger.info(f"Cache hit for {cache_key}")
                    if isinstance(cached_data, bytes):
                        try:
                            cached_data = cached_data.decode('utf-8')
                        except UnicodeDecodeError as e:
                            logger.error(f"Failed to decode cached data: {e}")
                            redis_client.delete(cache_key)
                            logger.warning(f"Deleted invalid cache {cache_key}")
                    return json.loads(cached_data)
            except redis.RedisError as e:
                logger.error(f"Redis error while caching data: {e}")

        if market == "KR":
            # pykrx로 한국 주식 티커 (상위 50개로 제한)
            tickers = stock.get_market_ticker_list(market="KOSPI")
            data = [{"ticker": t, "name": stock.get_market_ticker_name(t)} for t in tickers[:50]]
        else:
            # US 인기 종목 예시
            data = [
                {"ticker": "AAPL", "name": "Apple Inc."},
                {"ticker": "MSFT", "name": "Microsoft Corp."},
                {"ticker": "GOOGL", "name": "Alphabet Inc."},
                {"ticker": "AMZN", "name": "Amazon.com Inc."},
                {"ticker": "TSLA", "name": "Tesla Inc."},
            ]

        result = {"tickers": data}
        if redis_client:
            try:
                redis_client.setex(cache_key, 1800, json.dumps(result))
                logger.info(f"Cached {cache_key}")
            except redis.RedisError as e:
                logger.error(f"Redis error while caching data: {e}")

        return result

    except Exception as e:
        logger.error(f"❌ 티커 데이터 가져오기 오류: {e}")
        return {"tickers": []}

def calculate_indicators(df: pd.DataFrame) -> dict:
    """python-ta로 지표 계산 (TA-Lib 대체)"""
    if len(df) < 30:  # 최소 데이터 부족 시 None 반환
        return {k: None for k in ['RSI', 'MACD', 'CCI', 'MFI', 'ADX', 'SlowK', 'SlowD', 'SMA10', 'SMA50', 'EMA20', 'EMA50', 'BB_upper', 'BB_lower']}

    # 컬럼 확인 및 정리
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = df['Close']  # 기본값으로 Close 사용

    # 모든 TA 피처 추가
    df_ta = add_all_ta_features(
        df[required_cols],
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )

    # 최신값 추출 (NaN → None)
    latest = df_ta.iloc[-1]
    prev = df_ta.iloc[-2] if len(df_ta) > 1 else latest

    indicators = {
        'RSI': latest.get('momentum_rsi', None),
        'MACD': latest.get('trend_macd', None),
        'MACD_prev': prev.get('trend_macd', None),
        'CCI': latest.get('trend_cci', None),  # CCI는 trend_cci로 대체
        'MFI': latest.get('volume_mfi', None),
        'ADX': latest.get('trend_adx', None),
        'SlowK': latest.get('momentum_stoch', None),
        'SlowD': latest.get('momentum_stoch_signal', None),
        'SMA10': latest.get('trend_sma_fast', None),  # SMA10
        'SMA50': latest.get('trend_sma_slow', None),  # SMA50
        'EMA20': latest.get('trend_ema_fast', None),  # EMA20
        'EMA50': latest.get('trend_ema_slow', None),  # EMA50
        'BB_upper': latest.get('volatility_bbh', None),  # Bollinger Upper
        'BB_lower': latest.get('volatility_bbl', None),  # Bollinger Lower
    }

    # NaN 처리
    for k, v in indicators.items():
        if pd.isna(v):
            indicators[k] = None

    return indicators

@app.get("/analyze")
async def fetch_stock_data(ticker: str, market: str = "US", period: str = "3mo"):
    try:
        cache_key = f"analyze:{ticker}:{market}:{period}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {cache_key}")
                return json.loads(cached)

        # 데이터 가져오기
        if market == "KR":
            end_date = datetime.today().strftime("%Y%m%d")
            start_date = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
            df = df.rename(columns={
                "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"
            })
        else:
            df = yf.download(ticker, period=period, interval="1d")

        if df.empty:
            raise ValueError("No data found")

        df = df.dropna().reset_index(drop=True)

        # 지표 계산
        indicators = calculate_indicators(df)

        # 결과
        dates = df.index.astype(str).tolist()
        closes = df["Close"].round(2).tolist()
        volumes = df["Volume"].fillna(0).astype(float).tolist()

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
        logger.error(f"analyze error: {e}")
        return {"error": str(e)}

@app.get("/news")
async def fetch_news(ticker: str, market: str = "KR", start: int = 1, display: int = 10):
    try:
        if redis_client:
            cache_key = f"news:{ticker}:{market}:{start}:{display}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

        query = ticker if market == "KR" else f"{ticker} stock"
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID", ""),
            "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET", ""),
        }
        params = {"query": query, "display": display, "start": start, "sort": "date"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        news_list = []
        positive_keywords = ["상승", "호재", "강세", "매수"]
        negative_keywords = ["하락", "악재", "약세", "매도"]
        for item in items:
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            url_link = item.get("link", "")
            pub_date = item.get("pubDate", "")
            sentiment = "중립"
            title_lower = title.lower()
            if any(kw in title_lower for kw in positive_keywords):
                sentiment = "긍정"
            elif any(kw in title_lower for kw in negative_keywords):
                sentiment = "부정"
            color = {"긍정": "green", "부정": "red", "중립": "yellow"}.get(sentiment, "yellow")
            news_list.append({
                "title": title,
                "url": url_link,
                "date": pub_date,
                "sentiment": sentiment,
                "color": color
            })

        result = {"news": news_list, "total": data.get("total", 0)}
        if redis_client:
            redis_client.setex(cache_key, 1800, json.dumps(result))
        return result

    except Exception as e:
        logger.error(f"News fetch error: {e}")
        return {"news": [], "total": 0}

@app.get("/compare")
async def fetch_change_comparison(ticker: str, market: str = "US"):
    try:
        tickers_to_compare = [ticker]
        if market == "KR":
            tickers_to_compare.append("^KS11")  # KOSPI
        else:
            tickers_to_compare.append("^IXIC")  # NASDAQ

        changes = {}
        for t in tickers_to_compare:
            if t == ticker and market == "KR":
                end_date = datetime.today().strftime("%Y%m%d")
                start_date = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
                df = stock.get_market_ohlcv_by_date(start_date, end_date, t)
                df = df.rename(columns={"종가": "Close"})
                df["Close"] = df["Close"].astype(float)
            else:
                df = yf.download(t, period="3mo", interval="1d")

            if df.empty:
                changes[t] = 0.0
                continue

            first_close = df["Close"].iloc[0]
            last_close = df["Close"].iloc[-1]
            change = ((last_close - first_close) / first_close) * 100
            changes[t] = round(float(change), 2)

        result = {
            ticker: changes.get(ticker, 0.0),
            "KOSPI" if market == "KR" else "NASDAQ": changes.get("^KS11" if market == "KR" else "^IXIC", 0.0)
        }

        return result

    except Exception as e:
        logger.error(f"❌ 변동률 비교 중 오류 발생: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/ticker_info")
async def fetch_ticker_info(ticker: str, market: str = "US"):
    try:
        if market == "KR":
            name = stock.get_market_ticker_name(ticker)
            return {"name": name or ticker}
        else:
            info = yf.Ticker(ticker).info
            return {"name": info.get("longName", ticker)}
    except Exception as e:
        logger.error(f"❌ 종목 정보 요청 실패: {e}")
        return {"name": ticker}

@app.get("/health")
async def check_server_status():
    try:
        if redis_client:
            redis_client.ping()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}