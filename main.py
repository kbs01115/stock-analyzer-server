# main.py (간략 버전 - 성공용)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "서버 정상 실행 중!"}

@app.get("/tickers")
async def get_tickers(market: str = "KR"):
    if market == "KR":
        return {"tickers": [
            {"ticker": "005930", "name": "삼성전자"},
            {"ticker": "000660", "name": "SK하이닉스"},
        ]}
    else:
        return {"tickers": [
            {"ticker": "AAPL", "name": "Apple"},
            {"ticker": "MSFT", "name": "Microsoft"},
        ]}

@app.get("/health")
async def health():
    return {"status": "ok"}