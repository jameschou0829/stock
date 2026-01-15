"""
tools/check_signals_coverage.py

用途：找出「哪些交易日還沒 build_signals（或 build 不完整）」。

原理：
- 以某一檔「交易日 proxy」的 stock_daily 日期當作交易日曆（預設 settings.MARKET_PROXY_STOCK_ID=0050）
- 比對 stock_signals_v2 在該日的 distinct stock_id 數量
- 若 count=0 => 未 build
- 若 count < min_stocks（預設 settings.SIGNALS_MIN_STOCKS）=> 可能 build 不完整（或當天資料不完整）

用法：
  python tools/check_signals_coverage.py --start 2025-01-01 --end 2026-01-13
  python tools/check_signals_coverage.py --days 120
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import settings
from utils.db import db_conn


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=None, help="只看最近 N 天（與 --start/--end 擇一）")
    ap.add_argument("--calendar-stock-id", default=getattr(settings, "MARKET_PROXY_STOCK_ID", "0050"))
    ap.add_argument("--min-stocks", type=int, default=getattr(settings, "SIGNALS_MIN_STOCKS", 2000))
    args = ap.parse_args()

    if args.days is None and (not args.start or not args.end):
        raise SystemExit("請提供 --days N 或 --start YYYY-MM-DD --end YYYY-MM-DD")

    if args.days is not None:
        end = date.today()
        start = end - timedelta(days=int(args.days))
    else:
        start = parse_date(args.start)
        end = parse_date(args.end)

    cal_id = str(args.calendar_stock_id)
    min_stocks = int(args.min_stocks)

    with db_conn() as conn:
        with conn.cursor() as cur:
            # 交易日曆：用 proxy 股的日K日期
            cur.execute(
                """
                SELECT DISTINCT trading_date
                FROM stock_daily
                WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (cal_id, start, end),
            )
            days = [r["trading_date"] for r in cur.fetchall()]

            if not days:
                print(f"[WARN] 在 stock_daily 找不到 {cal_id} 的交易日資料：{start}~{end}")
                return

            # 每日 signals 覆蓋數
            missing = []
            incomplete = []
            ok = []
            for d in days:
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT stock_id) AS n
                    FROM stock_signals_v2
                    WHERE trading_date=%s
                    """,
                    (d,),
                )
                n = int(cur.fetchone()["n"] or 0)
                if n == 0:
                    missing.append((d, n))
                elif n < min_stocks:
                    incomplete.append((d, n))
                else:
                    ok.append((d, n))

    print(f"[OK] calendar_stock_id={cal_id} range={start}~{end} days={len(days)} min_stocks={min_stocks}")
    print(f"[OK] built_ok={len(ok)} missing={len(missing)} incomplete={len(incomplete)}")

    if missing:
        print("\n=== 未 build（signals=0）===")
        for d, n in missing[:200]:
            print(d, "stocks=", n)
        if len(missing) > 200:
            print(f"... (還有 {len(missing)-200} 天)")

    if incomplete:
        print("\n=== 可能不完整（signals < min_stocks）===")
        for d, n in incomplete[:200]:
            print(d, "stocks=", n)
        if len(incomplete) > 200:
            print(f"... (還有 {len(incomplete)-200} 天)")


if __name__ == "__main__":
    main()

