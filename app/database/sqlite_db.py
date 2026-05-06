import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_PATH = "db/adv_ai.db"

def get_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"DB 연결 에러: {e}")
        return None

def init_db():
    query_price = """
    CREATE TABLE IF NOT EXISTS stock_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,   -- 티커명
        date TEXT NOT NULL,     -- 날짜  
        open REAL,              -- 시가(장 시작 시 가격)
        high REAL,              -- 고가
        low REAL,               -- 저가
        close REAL,             -- 종가
        adj_close REAL,         -- 수정 종가(액면분할, 배당 등을 반영한 종가)
        volume INTEGER,         -- 거래량
        change_rate REAL,       -- 등락률
        nasdaq_close REAL,          -- 나스닥 종가
        nasdaq_change_rate REAL,    -- 나스닥 등락률
        alpha REAL,           -- 시장 대비 초과수익률 (Rm - Rs)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date) -- 동일 종목, 동일 날짜 데이터 중복 방지
    );
    """
    conn = get_connection()
    if conn:
        with conn:
            conn.execute(query_price)
        print("SQLite 테이블 초기화 완료")
        conn.close()

def save_price_to_db(df:pd.DataFrame):
    with get_connection() as conn:
        if conn is None:
            print("DB 연결 실패")
            return
        
        try:
            df.to_sql('stock_prices', conn, if_exists='append', index=False, chunksize=1000)
            print(f"DB 저장 성공: {len(df)}건")
        except Exception as e:
            print(f"DB 저장 에러: {e}")
    
