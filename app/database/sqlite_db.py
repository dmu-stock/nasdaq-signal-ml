import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_PATH = "db/adv_ai.db"

def get_connection():
    try:
        # db/ 디렉토리 자동 생성
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # FK 활성화
        conn.execute("PRAGMA foreign_keys = ON;")
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
        vix REAL,               -- VIX 공포지수
        tnx REAL,               -- 미국 10년물 국채금리
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date) -- 동일 종목, 동일 날짜 데이터 중복 방지
    );
    """

    # ── Member 테이블 ──
    # Discord 사용자 1명 = 1행. discord_id 는 Discord 의 user.id (snowflake) 그대로.
    query_member = """
    CREATE TABLE IF NOT EXISTS member (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        discord_id  TEXT NOT NULL UNIQUE,   -- Discord user ID (string)
        username    TEXT,                   -- 표시명 (참고용)
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # ── MemberStock 테이블 (워치리스트 + 포트폴리오) ──
    # quantity   : 보유 수량 (0 이면 워치리스트 전용 — 가격만 추적)
    # avg_buy_price: 평균 매수가 (NULL 이면 미설정 — P&L 계산 안 함)
    query_member_stock = """
    CREATE TABLE IF NOT EXISTS member_stock (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        member_id     INTEGER NOT NULL,
        ticker        TEXT NOT NULL,
        quantity      REAL NOT NULL DEFAULT 0,
        avg_buy_price REAL,
        added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(member_id, ticker),
        FOREIGN KEY (member_id) REFERENCES member(id) ON DELETE CASCADE
    );
    """

    conn = get_connection()
    if conn:
        with conn:
            conn.execute(query_price)
            conn.execute(query_member)
            conn.execute(query_member_stock)

            # 기존 DB 마이그레이션 — quantity / avg_buy_price / updated_at 없으면 ALTER
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(member_stock)").fetchall()
            }
            if "quantity" not in existing_cols:
                conn.execute(
                    "ALTER TABLE member_stock ADD COLUMN quantity REAL NOT NULL DEFAULT 0"
                )
                print("  → quantity 컬럼 추가")
            if "avg_buy_price" not in existing_cols:
                conn.execute(
                    "ALTER TABLE member_stock ADD COLUMN avg_buy_price REAL"
                )
                print("  → avg_buy_price 컬럼 추가")
            if "updated_at" not in existing_cols:
                conn.execute(
                    "ALTER TABLE member_stock ADD COLUMN updated_at TIMESTAMP"
                )
                print("  → updated_at 컬럼 추가")
        print("SQLite 테이블 초기화 완료 (stock_prices, member, member_stock)")
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
    
