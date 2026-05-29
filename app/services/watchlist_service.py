"""Member / MemberStock 워치리스트 CRUD.

Discord 봇의 `/c.내주식.*` 명령이 호출하는 데이터 레이어.
member 는 Discord user.id 를 키로 자동 생성 (회원가입 없이).
"""
from __future__ import annotations

from typing import Optional

from app.database.sqlite_db import get_connection


# ══════════════════════════════════════
# Member
# ══════════════════════════════════════
def get_or_create_member(discord_id: str, username: Optional[str] = None) -> int:
    """discord_id 로 member 를 찾거나 새로 생성. id (PK) 반환."""
    discord_id = str(discord_id).strip()
    if not discord_id:
        raise ValueError("discord_id is required")

    conn = get_connection()
    if conn is None:
        raise RuntimeError("DB 연결 실패")
    try:
        with conn:
            row = conn.execute(
                "SELECT id FROM member WHERE discord_id = ?",
                (discord_id,),
            ).fetchone()
            if row:
                # username 이 새로 들어왔으면 업데이트
                if username:
                    conn.execute(
                        "UPDATE member SET username = ? WHERE id = ?",
                        (username, row["id"]),
                    )
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO member (discord_id, username) VALUES (?, ?)",
                (discord_id, username),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


# ══════════════════════════════════════
# Watchlist (member_stock) CRUD
# ══════════════════════════════════════
def list_watchlist(discord_id: str) -> list[dict]:
    """워치리스트 ticker 목록 + 수량/평단 (최신 추가 순)."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            SELECT ms.ticker, ms.quantity, ms.avg_buy_price, ms.added_at, ms.updated_at
              FROM member_stock ms
              JOIN member m ON m.id = ms.member_id
             WHERE m.discord_id = ?
          ORDER BY ms.added_at DESC
            """,
            (str(discord_id),),
        ).fetchall()
        return [
            {
                "ticker": r["ticker"],
                "quantity": float(r["quantity"] or 0),
                "avg_buy_price": (float(r["avg_buy_price"]) if r["avg_buy_price"] is not None else None),
                "added_at": r["added_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def add_to_watchlist(
    discord_id: str,
    ticker: str,
    username: Optional[str] = None,
    quantity: Optional[float] = None,
    avg_buy_price: Optional[float] = None,
) -> dict:
    """ticker 를 워치리스트에 추가.

    - quantity 미지정 → 0 (워치리스트 전용)
    - avg_buy_price 미지정 → NULL (P&L 계산 안 함)
    - 이미 있으면 quantity/avg_buy_price 가 들어왔을 때만 업데이트, 아니면 already=True
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "error": "ticker is empty"}
    if quantity is not None and quantity < 0:
        return {"ok": False, "error": "quantity 는 0 이상이어야 해요"}
    if avg_buy_price is not None and avg_buy_price <= 0:
        return {"ok": False, "error": "avg_buy_price 는 양수여야 해요"}

    member_id = get_or_create_member(discord_id, username=username)
    conn = get_connection()
    if conn is None:
        return {"ok": False, "error": "DB 연결 실패"}
    try:
        with conn:
            existing = conn.execute(
                "SELECT id, quantity, avg_buy_price FROM member_stock WHERE member_id = ? AND ticker = ?",
                (member_id, ticker),
            ).fetchone()
            if existing:
                # 이미 있고 새 값이 들어왔으면 업데이트
                updated_fields = []
                if quantity is not None:
                    updated_fields.append(("quantity", float(quantity)))
                if avg_buy_price is not None:
                    updated_fields.append(("avg_buy_price", float(avg_buy_price)))
                if updated_fields:
                    set_clause = ", ".join(f"{f} = ?" for f, _ in updated_fields)
                    set_clause += ", updated_at = CURRENT_TIMESTAMP"
                    conn.execute(
                        f"UPDATE member_stock SET {set_clause} WHERE id = ?",
                        [v for _, v in updated_fields] + [existing["id"]],
                    )
                    return {
                        "ok": True, "already": True, "updated": True, "ticker": ticker,
                        "quantity": float(quantity if quantity is not None else existing["quantity"] or 0),
                        "avg_buy_price": float(avg_buy_price) if avg_buy_price is not None else (
                            float(existing["avg_buy_price"]) if existing["avg_buy_price"] is not None else None
                        ),
                    }
                return {"ok": True, "already": True, "updated": False, "ticker": ticker}

            # 신규 INSERT
            conn.execute(
                """
                INSERT INTO member_stock (member_id, ticker, quantity, avg_buy_price)
                VALUES (?, ?, ?, ?)
                """,
                (member_id, ticker, float(quantity or 0), avg_buy_price),
            )
            return {
                "ok": True, "already": False, "ticker": ticker,
                "quantity": float(quantity or 0),
                "avg_buy_price": float(avg_buy_price) if avg_buy_price is not None else None,
            }
    finally:
        conn.close()


def update_watchlist_item(
    discord_id: str,
    ticker: str,
    quantity: Optional[float] = None,
    avg_buy_price: Optional[float] = None,
) -> dict:
    """기존 항목의 수량 또는 평단가 업데이트.

    둘 다 None 이면 변경 없음 (no-op). ticker 가 없으면 found=False.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "error": "ticker is empty"}
    if quantity is None and avg_buy_price is None:
        return {"ok": False, "error": "수량 또는 평단가 중 하나는 지정해야 해요"}
    if quantity is not None and quantity < 0:
        return {"ok": False, "error": "quantity 는 0 이상이어야 해요"}
    if avg_buy_price is not None and avg_buy_price <= 0:
        return {"ok": False, "error": "avg_buy_price 는 양수여야 해요"}

    conn = get_connection()
    if conn is None:
        return {"ok": False, "error": "DB 연결 실패"}
    try:
        with conn:
            existing = conn.execute(
                """
                SELECT ms.id FROM member_stock ms
                  JOIN member m ON m.id = ms.member_id
                 WHERE m.discord_id = ? AND ms.ticker = ?
                """,
                (str(discord_id), ticker),
            ).fetchone()
            if not existing:
                return {"ok": True, "found": False, "ticker": ticker}

            updated_fields = []
            if quantity is not None:
                updated_fields.append(("quantity", float(quantity)))
            if avg_buy_price is not None:
                updated_fields.append(("avg_buy_price", float(avg_buy_price)))
            set_clause = ", ".join(f"{f} = ?" for f, _ in updated_fields)
            set_clause += ", updated_at = CURRENT_TIMESTAMP"
            conn.execute(
                f"UPDATE member_stock SET {set_clause} WHERE id = ?",
                [v for _, v in updated_fields] + [existing["id"]],
            )
            return {
                "ok": True, "found": True, "ticker": ticker,
                "quantity": quantity, "avg_buy_price": avg_buy_price,
            }
    finally:
        conn.close()


def remove_from_watchlist(discord_id: str, ticker: str) -> dict:
    """ticker 를 워치리스트에서 제거. 없으면 found=False."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ok": False, "error": "ticker is empty"}

    conn = get_connection()
    if conn is None:
        return {"ok": False, "error": "DB 연결 실패"}
    try:
        with conn:
            row = conn.execute(
                """
                SELECT ms.id FROM member_stock ms
                  JOIN member m ON m.id = ms.member_id
                 WHERE m.discord_id = ? AND ms.ticker = ?
                """,
                (str(discord_id), ticker),
            ).fetchone()
            if not row:
                return {"ok": True, "found": False, "ticker": ticker}
            conn.execute("DELETE FROM member_stock WHERE id = ?", (row["id"],))
            return {"ok": True, "found": True, "ticker": ticker}
    finally:
        conn.close()


def clear_watchlist(discord_id: str) -> dict:
    """워치리스트 전체 삭제. 삭제 건수 반환."""
    conn = get_connection()
    if conn is None:
        return {"ok": False, "error": "DB 연결 실패"}
    try:
        with conn:
            cur = conn.execute(
                """
                DELETE FROM member_stock
                 WHERE member_id IN (SELECT id FROM member WHERE discord_id = ?)
                """,
                (str(discord_id),),
            )
            return {"ok": True, "deleted": cur.rowcount}
    finally:
        conn.close()
