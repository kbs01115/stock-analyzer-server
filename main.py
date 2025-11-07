# 파일명: main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from fastapi.responses import JSONResponse
import requests
from pykrx import stock
from datetime import datetime, timedelta
import json
import redis
import logging

    
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 한글 종목명을 영어로 변환하기 위한 매핑 테이블
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
                            logger.info(f"Deleted invalid cache for {cache_key}")
                            cached_data = None
                    if cached_data:
                        return JSONResponse(content=json.loads(cached_data), media_type="application/json; charset=utf-8")
            except (redis.RedisError, json.JSONDecodeError) as e:
                logger.error(f"Redis cache fetch error: {e}")
                try:
                    redis_client.delete(cache_key)
                    logger.info(f"Deleted invalid cache for {cache_key}")
                except redis.RedisError as e:
                    logger.error(f"Redis delete error: {e}")

        # 데이터 가져오기
        if market == "KR":
            try:
                kospi_tickers = stock.get_market_ticker_list(market="KOSPI")
                kosdaq_tickers = stock.get_market_ticker_list(market="KOSDAQ")
            except Exception as e:
                logger.error(f"pykrx 데이터 가져오기 실패: {e}")
                kospi_tickers, kosdaq_tickers = [], []
            tickers = kospi_tickers + kosdaq_tickers
            ticker_names = [
                {"ticker": ticker, "name": stock.get_market_ticker_name(ticker) or ticker}
                for ticker in tickers
            ]
        else:
            # US 시장 티커 목록 확장
            try:
                # yfinance로 S&P 500 종목 목록 가져오기 (예시)
                sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                ticker_names = [
                    {"ticker": row['Symbol'], "name": row['Security']}
                    for _, row in sp500.iterrows()
                ]
            except Exception as e:
                logger.error(f"US 티커 목록 가져오기 실패: {e}")
                # 기본 인기 종목으로 대체
                ticker_names = [
                    {"ticker": "AAPL", "name": "Apple Inc."},
                    {"ticker": "MSFT", "name": "Microsoft Corporation"},
                    {"ticker": "TSLA", "name": "Tesla Inc."},
                    {"ticker": "GOOGL", "name": "Alphabet Inc."},
                    {"ticker": "AMZN", "name": "Amazon.com Inc."},
                ]

        # Redis에 캐싱 (TTL: 1시간)
        if redis_client:
            try:
                redis_client.setex(cache_key, 3600, json.dumps({"tickers": ticker_names}))
                logger.info(f"Data cached for {cache_key}")
            except redis.RedisError as e:
                logger.error(f"Redis error while caching data: {e}")

        return JSONResponse(content={"tickers": ticker_names}, media_type="application/json; charset=utf-8")
    except Exception as e:
        logger.error(f"❌ 티커 목록 가져오기 오류: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/ticker_info")
async def get_ticker_info(ticker: str, market: str = "US"):
    try:
        # Redis 캐시 확인
        if redis_client:
            cache_key = f"ticker_info:{market}:{ticker}"
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
                            logger.info(f"Deleted invalid cache for {cache_key}")
                            cached_data = None
                    if cached_data:
                        return json.loads(cached_data)
            except (redis.RedisError, json.JSONDecodeError) as e:
                logger.error(f"Redis cache fetch error: {e}")
                try:
                    redis_client.delete(cache_key)
                    logger.info(f"Deleted invalid cache for {cache_key}")
                except redis.RedisError as e:
                    logger.error(f"Redis delete error: {e}")

        # 데이터 가져오기
        if market == "KR":
            try:
                name = stock.get_market_ticker_name(ticker)
                result = {"ticker": ticker, "name": name or ticker}
            except Exception as e:
                logger.error(f"pykrx 데이터 가져오기 실패: {e}")
                result = {"ticker": ticker, "name": ticker}
        else:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                if not info or "symbol" not in info:
                    logger.error(f"Invalid Ticker: {ticker}")
                    return JSONResponse(status_code=400, content={"error": f"유효하지 않은 티커: {ticker}"})
                name = info.get("longName", ticker)
                result = {"ticker": ticker, "name": name}
            except Exception as e:
                logger.error(f"yfinance 데이터 가져오기 실패: {e}")
                result = {"ticker": ticker, "name": ticker}

        # Redis에 캐싱 (TTL: 1일)
        if redis_client:
            try:
                redis_client.setex(cache_key, 86400, json.dumps(result))
                logger.info(f"Data cached for {cache_key}")
            except redis.RedisError as e:
                logger.error(f"Redis error while caching data: {e}")

        return result
    except Exception as e:
        logger.error(f"❌ 티커 정보 가져오기 오류: {e}")
        return JSONResponse(status_code=500, content={"ticker": ticker, "name": ticker})

@app.get("/analyze")
async def analyze(ticker: str, market: str = "US", period: str = "3mo"):
    try:
        # Redis 캐시 확인
        if redis_client:
            cache_key = f"analyze:{market}:{ticker}:{period}"
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
                            logger.info(f"Deleted invalid cache for {cache_key}")
                            cached_data = None
                    if cached_data:
                        return json.loads(cached_data)
            except (redis.RedisError, json.JSONDecodeError) as e:
                logger.error(f"Redis cache fetch error: {e}")
                try:
                    redis_client.delete(cache_key)
                    logger.info(f"Deleted invalid cache for {cache_key}")
                except redis.RedisError as e:
                    logger.error(f"Redis delete error: {e}")

        # 데이터 가져오기
        if market == "KR":
            end_date = datetime.today().strftime("%Y%m%d")
            start_date = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
            try:
                df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
                df = df.rename(columns={
                    "시가": "Open",
                    "고가": "High",
                    "저가": "Low",
                    "종가": "Close",
                    "거래량": "Volume"
                })
                df.index.name = "Date"
                df.index = pd.to_datetime(df.index)
                df["Close"] = df["Close"].astype(int)
            except Exception as e:
                logger.error(f"pykrx 데이터 가져오기 실패: {e}")
                return JSONResponse(status_code=400, content={"error": f"데이터 가져오기 실패: {ticker}"})
        else:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                if not info or "symbol" not in info:
                    logger.error(f"Invalid Ticker: {ticker}")
                    return JSONResponse(status_code=400, content={"error": f"유효하지 않은 티커: {ticker}"})
                
                df = yf.download(ticker, period=period, interval="1d")
                logger.info(f"US Market - Ticker: {ticker}, Data Preview: {df.head()}")
                logger.info(f"US Market - Data Length: {len(df)}")
            except Exception as e:
                logger.error(f"yfinance 데이터 가져오기 실패: {e}")
                return JSONResponse(status_code=400, content={"error": f"데이터 가져오기 실패: {ticker}"})

        if df.empty:
            logger.error(f"US Market - Empty Data for Ticker: {ticker}")
            return JSONResponse(status_code=400, content={"error": f"데이터가 비어 있습니다. 티커 {ticker}를 확인하세요."})

        logger.info(f"US Market - Columns: {df.columns}, Index Type: {type(df.index)}")

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            else:
                return JSONResponse(status_code=400, content={"error": "Date 컬럼이 없습니다."})

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not isinstance(df.index, pd.DatetimeIndex):
            return JSONResponse(status_code=400, content={"error": "인덱스가 DateTimeIndex 형식이 아닙니다."})

        required_columns = ["Close", "High", "Low", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns for {ticker}: {missing_columns}")
            return JSONResponse(status_code=400, content={"error": f"필수 컬럼 {missing_columns}이 누락되었습니다."})

        df = df.dropna()
        if df.empty:
            return JSONResponse(status_code=400, content={"error": "NaN 값을 제거한 후 데이터가 비어 있습니다."})

        df.loc[df["Volume"] == 0, "Volume"] = 1e-6
        df["Volume"] = df["Volume"].astype(np.float64)

        logger.info(f"데이터 길이: {len(df)}")
        if len(df) < 50:
            logger.warning(f"데이터 길이 부족: {len(df)}일 (최소 50일 권장)")

        closes = df["Close"].values.tolist()
        if market == "KR":
            closes = [int(x) for x in closes]
        volumes = df["Volume"].values.tolist()
        dates = df.index.strftime("%Y-%m-%d").tolist()
        logger.info(f"종가 데이터: {closes[:5]}...")
        logger.info(f"거래량 데이터: {volumes[:5]}...")
        logger.info(f"날짜 데이터: {dates[:5]}...")

        close = np.array(df["Close"].values, dtype=np.float64)
        high = np.array(df["High"].values, dtype=np.float64)
        low = np.array(df["Low"].values, dtype=np.float64)
        volume = df["Volume"].values.astype(np.float64)

        logger.info(f"Close 길이: {len(close)}, High 길이: {len(high)}, Low 길이: {len(low)}, Volume 길이: {len(volume)}")

        for arr, name in [(close, "Close"), (high, "High"), (low, "Low"), (volume, "Volume")]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return JSONResponse(status_code=400, content={"error": f"{name} 데이터에 NaN 또는 inf 값이 포함되어 있습니다."})

        indicators = {}
        indicators_series = {}

        try:
            rsi = talib.RSI(close, timeperiod=7)
            indicators["RSI"] = float(rsi[-1]) if len(close) >= 7 else None
            indicators_series["RSI"] = [float(x) if not np.isnan(x) else None for x in rsi] if len(close) >= 7 else []
        except Exception as e:
            logger.error(f"RSI 계산 중 오류: {e}")
            indicators["RSI"] = None
            indicators_series["RSI"] = []

        try:
            macd = talib.MACD(close, fastperiod=6, slowperiod=13, signalperiod=5)
            indicators["MACD"] = float(macd[0][-1]) if len(close) >= 13 else None
            indicators["MACD_prev"] = float(macd[0][-2]) if len(close) >= 13 else None
            indicators_series["MACD"] = [float(x) if not np.isnan(x) else None for x in macd[0]] if len(close) >= 13 else []
        except Exception as e:
            logger.error(f"MACD 계산 중 오류: {e}")
            indicators["MACD"] = None
            indicators["MACD_prev"] = None
            indicators_series["MACD"] = []

        try:
            cci = talib.CCI(high, low, close, timeperiod=7)
            indicators["CCI"] = float(cci[-1]) if len(close) >= 7 else None
            indicators_series["CCI"] = [float(x) if not np.isnan(x) else None for x in cci] if len(close) >= 7 else []
        except Exception as e:
            logger.error(f"CCI 계산 중 오류: {e}")
            indicators["CCI"] = None
            indicators_series["CCI"] = []

        try:
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)
            indicators["MFI"] = float(mfi[-1]) if len(close) >= 14 else None
            indicators_series["MFI"] = [float(x) if not np.isnan(x) else None for x in mfi] if len(close) >= 14 else []
        except Exception as e:
            logger.error(f"MFI 계산 중 오류: {e}")
            indicators["MFI"] = None
            indicators_series["MFI"] = []

        try:
            adx = talib.ADX(high, low, close, timeperiod=7)
            indicators["ADX"] = float(adx[-1]) if len(close) >= 7 else None
            indicators_series["ADX"] = [float(x) if not np.isnan(x) else None for x in adx] if len(close) >= 7 else []
        except Exception as e:
            logger.error(f"ADX 계산 중 오류: {e}")
            indicators["ADX"] = None
            indicators_series["ADX"] = []

        try:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=7, slowk_period=3, slowd_period=3)
            indicators["SlowK"] = float(slowk[-1]) if len(close) >= 7 else None
            indicators["SlowD"] = float(slowd[-1]) if len(close) >= 7 else None
            indicators["SlowK_prev"] = float(slowk[-2]) if len(close) >= 7 else None
            indicators["SlowD_prev"] = float(slowd[-2]) if len(close) >= 7 else None
            indicators_series["SlowK"] = [float(x) if not np.isnan(x) else None for x in slowk] if len(close) >= 7 else []
            indicators_series["SlowD"] = [float(x) if not np.isnan(x) else None for x in slowd] if len(close) >= 7 else []
        except Exception as e:
            logger.error(f"STOCH 계산 중 오류: {e}")
            indicators["SlowK"] = None
            indicators["SlowD"] = None
            indicators["SlowK_prev"] = None
            indicators["SlowD_prev"] = None
            indicators_series["SlowK"] = []
            indicators_series["SlowD"] = []

        try:
            sma10 = talib.SMA(close, timeperiod=10)
            sma50 = talib.SMA(close, timeperiod=50)
            indicators["SMA10"] = float(sma10[-1]) if len(close) >= 10 else None
            indicators["SMA50"] = float(sma50[-1]) if len(close) >= 50 else None
            indicators["SMA10_prev"] = float(sma10[-2]) if len(close) >= 10 else None
            indicators["SMA50_prev"] = float(sma50[-2]) if len(close) >= 50 else None
            indicators_series["SMA10"] = [float(x) if not np.isnan(x) else None for x in sma10] if len(close) >= 10 else []
            indicators_series["SMA50"] = [float(x) if not np.isnan(x) else None for x in sma50] if len(close) >= 50 else []
        except Exception as e:
            logger.error(f"SMA 계산 중 오류: {e}")
            indicators["SMA10"] = None
            indicators["SMA50"] = None
            indicators["SMA10_prev"] = None
            indicators["SMA50_prev"] = None
            indicators_series["SMA10"] = []
            indicators_series["SMA50"] = []

        try:
            ema20 = talib.EMA(close, timeperiod=20)
            ema50 = talib.EMA(close, timeperiod=50)
            indicators["EMA20"] = float(ema20[-1]) if len(close) >= 20 else None
            indicators["EMA50"] = float(ema50[-1]) if len(close) >= 50 else None
            indicators["EMA20_prev"] = float(ema20[-2]) if len(close) >= 20 else None
            indicators["EMA50_prev"] = float(ema50[-2]) if len(close) >= 50 else None
            indicators_series["EMA20"] = [float(x) if not np.isnan(x) else None for x in ema20] if len(close) >= 20 else []
            indicators_series["EMA50"] = [float(x) if not np.isnan(x) else None for x in ema50] if len(close) >= 50 else []
        except Exception as e:
            logger.error(f"EMA 계산 중 오류: {e}")
            indicators["EMA20"] = None
            indicators["EMA50"] = None
            indicators["EMA20_prev"] = None
            indicators["EMA50_prev"] = None
            indicators_series["EMA20"] = []
            indicators_series["EMA50"] = []

        try:
            bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=14, nbdevup=1.8, nbdevdn=1.8)
            indicators["BB_upper"] = float(bb_upper[-1]) if len(close) >= 14 else None
            indicators["BB_lower"] = float(bb_lower[-1]) if len(close) >= 14 else None
            indicators["BB_upper_prev"] = float(bb_upper[-2]) if len(close) >= 14 else None
            indicators["BB_lower_prev"] = float(bb_lower[-2]) if len(close) >= 14 else None
            indicators_series["BB_upper"] = [float(x) if not np.isnan(x) else None for x in bb_upper] if len(close) >= 14 else []
            indicators_series["BB_lower"] = [float(x) if not np.isnan(x) else None for x in bb_lower] if len(close) >= 14 else []
        except Exception as e:
            logger.error(f"BBANDS 계산 중 오류: {e}")
            indicators["BB_upper"] = None
            indicators["BB_lower"] = None
            indicators["BB_upper_prev"] = None
            indicators["BB_lower_prev"] = None
            indicators_series["BB_upper"] = []
            indicators_series["BB_lower"] = []

        try:
            obv = talib.OBV(close, volume)
            indicators["OBV"] = float(obv[-1]) if len(close) >= 1 else None
            indicators["OBV_prev"] = float(obv[-2]) if len(close) >= 2 else None
        except Exception as e:
            logger.error(f"OBV 계산 중 오류: {e}")
            indicators["OBV"] = None
            indicators["OBV_prev"] = None

        try:
            atr = talib.ATR(high, low, close, timeperiod=14)
            indicators["ATR"] = float(atr[-1]) if len(close) >= 14 else None
        except Exception as e:
            logger.error(f"ATR 계산 중 오류: {e}")
            indicators["ATR"] = None

        indicators["Close"] = int(close[-1]) if len(close) >= 1 and market == "KR" else float(close[-1]) if len(close) >= 1 else None
        indicators["Close_prev"] = int(close[-2]) if len(close) >= 2 and market == "KR" else float(close[-2]) if len(close) >= 2 else None

        indicators = {
            k: round(v, 3) if v is not None and not np.isnan(v) and k not in ["Close", "Close_prev"] else v
            for k, v in indicators.items()
        }

        result = {
            "dates": dates,
            "closes": closes,
            "volumes": volumes,
            "indicators": indicators,
            "indicators_series": indicators_series
        }

        # Redis에 캐싱 (TTL: 5분)
        if redis_client:
            try:
                redis_client.setex(cache_key, 300, json.dumps(result))
                logger.info(f"Data cached for {cache_key}")
            except redis.RedisError as e:
                logger.error(f"Redis error while caching data: {e}")

        return result

    except Exception as e:
        logger.error(f"❌ 데이터 분석 중 오류 발생: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/news")
async def get_news(ticker: str, market: str = "US", start: int = 1, display: int = 10):
    try:
        if redis_client:
            cache_key = f"news:{market}:{ticker}:{start}:{display}"
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
                            logger.info(f"Deleted invalid cache for {cache_key}")
                            cached_data = None
                    if cached_data:
                        return json.loads(cached_data)
            except (redis.RedisError, json.JSONDecodeError) as e:
                logger.error(f"Redis cache fetch error: {e}")
                try:
                    redis_client.delete(cache_key)
                    logger.info(f"Deleted invalid cache for {cache_key}")
                except redis.RedisError as e:
                    logger.error(f"Redis delete error: {e}")

        query = ticker
        if market == "KR":
            try:
                korean_name = stock.get_market_ticker_name(ticker) or ticker
                query = f"{korean_name} 뉴스"
                logger.info(f"KR 종목명: {korean_name}, 검색 쿼리: {query}")
            except Exception as e:
                logger.error(f"pykrx 종목명 가져오기 실패: {e}")
                query = f"{ticker} 뉴스"

        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": "hGsBkHMZAIdA274Yf1HM",  # 네이버 개발자 센터에서 발급받은 키로 교체
            "X-Naver-Client-Secret": "DATh9ARioQ"  # 네이버 개발자 센터에서 발급받은 키로 교체
        }
        params = {
            "query": query,
            "display": min(display, 100),
            "start": start,
            "sort": "date"
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.error(f"❌ 네이버 뉴스 API 요청 실패: {response.status_code} - {response.text}")
            return {"news": [], "total": 0}

        try:
            data = response.json()
            logger.info(f"Naver API response: {data.get('items', [])[:2]}")
        except ValueError as e:
            logger.error(f"네이버 API 응답 파싱 실패: {e} - {response.text}")
            return {"news": [], "total": 0}

        articles = data.get('items', [])
        total = data.get('total', 0)
        if not articles:
            logger.warning(f"네이버 API에서 뉴스 데이터 없음: {data}")

        positive_keywords = ['상승', '호재', '성장', '기대', '성공', '수익', '좋은', '긍정', '회복', '강세', 'increase', 'positive', 'growth', 'success', 'profit']
        negative_keywords = ['하락', '악재', '침체', '우려', '부진', '손실', '나쁜', '부정', '약세', '위기', 'decrease', 'negative', 'decline', 'loss', 'crisis']

        news_list = [
            {
                "title": article.get('title', '제목 없음').replace('<b>', '').replace('</b>', '') if article.get('title') else '제목 없음',
                "link": article.get('link', ''),
                "pubDate": article.get('pubDate', ''),
                "sentiment": (
                    "긍정" if article.get('title') and any(keyword in article.get('title', '').lower() for keyword in positive_keywords) and
                    not any(keyword in article.get('title', '').lower() for keyword in negative_keywords) else
                    "부정" if article.get('title') and any(keyword in article.get('title', '').lower() for keyword in negative_keywords) and
                    not any(keyword in article.get('title', '').lower() for keyword in positive_keywords) else
                    "중립"
                ),
                "color": (
                    "#00FF00" if article.get('title') and any(keyword in article.get('title', '').lower() for keyword in positive_keywords) and
                    not any(keyword in article.get('title', '').lower() for keyword in negative_keywords) else
                    "#FF0000" if article.get('title') and any(keyword in article.get('title', '').lower() for keyword in negative_keywords) and
                    not any(keyword in article.get('title', '').lower() for keyword in positive_keywords) else
                    "#FFFF00"
                )
            }
            for article in articles
        ]

        if redis_client:
            try:
                redis_client.setex(cache_key, 1800, json.dumps({"news": news_list, "total": total}))
                logger.info(f"Data cached for {cache_key}")
            except redis.RedisError as e:
                logger.error(f"Redis error while caching data: {e}")

        return {"news": news_list, "total": total}

    except Exception as e:
        logger.error(f"❌ 뉴스 데이터 가져오기 오류: {e}")
        return {"news": [], "total": 0}

@app.get("/compare")
async def compare(ticker: str, market: str = "US"):
    try:
        tickers = [ticker, "^KS11", "^IXIC"]
        changes = {}

        for t in tickers:
            if t == ticker and market == "KR":
                end_date = datetime.today().strftime("%Y%m%d")
                start_date = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
                try:
                    df = stock.get_market_ohlcv_by_date(start_date, end_date, t)
                    df = df.rename(columns={"종가": "Close"})
                    df["Close"] = df["Close"].astype(int)
                except Exception as e:
                    logger.error(f"pykrx 데이터 가져오기 실패 for {t}: {e}")
                    changes[t] = 0.0
                    continue
            else:
                try:
                    df = yf.download(t, period="3mo", interval="1d")
                except Exception as e:
                    logger.error(f"yfinance 데이터 가져오기 실패 for {t}: {e}")
                    changes[t] = 0.0
                    continue
            
            if df.empty:
                logger.error(f"Empty data for {t} in compare endpoint")
                changes[t] = 0.0
                continue

            first_close = df["Close"].iloc[0]
            last_close = df["Close"].iloc[-1]
            change = ((last_close - first_close) / first_close) * 100
            changes[t] = round(float(change), 2)

        result = {
            ticker: changes[ticker],
            "KOSPI": changes["^KS11"],
            "NASDAQ": changes["^IXIC"]
        }

        return result

    except Exception as e:
        logger.error(f"❌ 변동률 비교 중 오류 발생: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})