"""
etl/backfill_turnover.py

目的：回填 stock_daily.turnover（成交金額），讓既有歷史資料可以產生教學版「量大動能」訊號。

規則：
- 只回填 turnover IS NULL 的資料（不覆蓋已存在的官方/其他來源成交金額）
- 若 turnover 為 NULL 且 close/volume 都有值 -> turnover = ROUND(close * volume)
- 分批更新（以 trading_date 區間切批），避免一次更新鎖表太久
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, date

# allow running as: python etl/backfill_turnover.py ...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import db_conn


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange_batches(start: date, end: date, batch_days: int):
    cur = start
    while cur <= end:
        b_end = min(end, cur + timedelta(days=batch_days - 1))
        yield cur, b_end
        cur = b_end + timedelta(days=1)


def daterange_batches_latest_first(start: date, end: date, batch_days: int):
    """
    由 end 往 start 反向切批（方便先補最近交易日）
    """
    cur_end = end
    while cur_end >= start:
        b_start = max(start, cur_end - timedelta(days=batch_days - 1))
        yield b_start, cur_end
        cur_end = b_start - timedelta(days=1)


def count_rows_to_update(conn, start: date, end: date) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM stock_daily
            WHERE trading_date BETWEEN %s AND %s
              AND turnover IS NULL
              AND close IS NOT NULL
              AND volume IS NOT NULL
            """,
            (start, end),
        )
        row = cur.fetchone()
        return int(row["cnt"] or 0) if row else 0


def backfill_batch(conn, start: date, end: date) -> int:
    """
    回傳本批次實際更新筆數（rowcount）
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE stock_daily
            SET turnover = ROUND(close * volume)
            WHERE trading_date BETWEEN %s AND %s
              AND turnover IS NULL
              AND close IS NOT NULL
              AND volume IS NOT NULL
            """,
            (start, end),
        )
        return int(cur.rowcount or 0)


def main():
    ap = argparse.ArgumentParser(description="Backfill stock_daily.turnover for NULL rows (close*volume).")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--batch-days", type=int, default=30, help="default=30")
    ap.add_argument("--latest-first", action="store_true", help="process batches from end -> start (newest first)")
    ap.add_argument("--dry-run", action="store_true", help="only print how many rows would be updated")
    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    batch_days = int(args.batch_days)
    if batch_days <= 0:
        raise ValueError("--batch-days must be > 0")
    if start > end:
        raise ValueError("--start must be <= --end")

    total = 0
    batch_iter = (
        daterange_batches_latest_first(start, end, batch_days)
        if args.latest_first
        else daterange_batches(start, end, batch_days)
    )
    for b_start, b_end in batch_iter:
        with db_conn(commit_on_success=not args.dry_run) as conn:
            cnt = count_rows_to_update(conn, b_start, b_end)
            if args.dry_run:
                print(f"[DRY-RUN] {b_start} ~ {b_end}: would update {cnt} rows")
                total += cnt
                continue

            if cnt == 0:
                print(f"[SKIP] {b_start} ~ {b_end}: 0 rows")
                continue

            updated = backfill_batch(conn, b_start, b_end)
            total += updated
            print(f"[OK] {b_start} ~ {b_end}: updated={updated} (eligible={cnt})")

    if args.dry_run:
        print(f"[DRY-RUN] total would update: {total}")
    else:
        print(f"[DONE] total updated: {total}")


if __name__ == "__main__":
    main()

