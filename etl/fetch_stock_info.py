# etl/fetch_stock_info.py

from utils.db import db_conn
from utils.fetcher import fetch_with_retry
from datetime import date, timedelta


def fetch_stock_info():
    data = fetch_with_retry(
        dataset="TaiwanStockInfo",
        params={},  # 不給日期會拿到最新一版
    )
    rows = data.get("data", [])
    print(f"股票清單 API 回傳：{len(rows)} 筆")

    # 過濾掉指數跟大盤
    filtered = [
        r for r in rows
        if r.get("industry_category") not in ("Index", "大盤")
    ]
    print(f"過濾後有效股票：{len(filtered)} 檔")

    return filtered

def is_active_stock(item: dict) -> int:
    stock_id = item["stock_id"]

    # 1. 指數 / 非股票
    if item["industry_category"] == "Index":
        return 0

    # 2. 非純數字（ETF、指標）
    if not stock_id.isdigit():
        return 0

    # 3. 沒有任何日K資料 → 視為無效
    if not has_recent_trading(stock_id):
        return 0

    return 1

def has_recent_trading(stock_id: str, days: int = 30) -> bool:
    """
    檢查最近 N 天內是否還有交易資料
    """
    with db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT MAX(trading_date) AS last_date
                FROM stock_daily
                WHERE stock_id = %s
                """,
                (stock_id,),
            )
            row = cursor.fetchone()
            if not row or not row["last_date"]:
                return False
            return row["last_date"] >= (date.today() - timedelta(days=days))

def save_stock_info(items):
    sql = """
    INSERT INTO stock_info
      (stock_id, stock_name, industry, type, ipo_date, is_active)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      stock_name = VALUES(stock_name),
      industry   = VALUES(industry),
      type       = VALUES(type),
      ipo_date   = VALUES(ipo_date),
      is_active  = VALUES(is_active)
    """

    rows = []
    for r in items:
        rows.append((
            r["stock_id"],
            r["stock_name"],
            r["industry_category"],
            r["type"],
            r["date"] if r["date"] else None,
            is_active_stock(r)
        ))

    if not rows:
        return

    with db_conn(commit_on_success=True) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql, rows)


def run_stock_info():
    items = fetch_stock_info()
    save_stock_info(items)
    print("📦 [OK] 股票清單維護完成！")
    print(f"有效股票：{len(items)} 檔")

if __name__ == "__main__":
    run_stock_info()
