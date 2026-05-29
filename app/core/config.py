"""환경변수 / 설정 로딩."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # 프로젝트 루트 (FastAPI/)
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embed_model: str
    chroma_dir: Path
    news_csv_path: Path  # LLM 분석 입력용 (data/news_*.csv)
    market: str  # "us" | "kr"


@lru_cache
def get_settings() -> Settings:
    s = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip(),
        chroma_dir=Path(os.getenv("CHROMA_DIR", str(ROOT / "chroma_db"))).resolve(),
        news_csv_path=Path(os.getenv("NEWS_CSV_PATH", str(ROOT / "data" / "news_2025.csv"))).resolve(),
        market=os.getenv("MARKET", "us").strip().lower(),
    )
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return s
