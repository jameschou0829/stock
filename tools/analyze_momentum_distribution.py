"""
tools/analyze_momentum_distribution.py

目的：驗證 momentum_score 的分布，確認 threshold=7.5 有統計意義。

需求：
1) 讀取 stock_signals_v2 最近 N 個交易日（預設 60）
2) 輸出 momentum_score 的統計：min/median/p90/p95/max
3) 輸出符合 momentum_score>7.5 的比例
4) 輸出符合 (momentum_score>7.5 AND yesterday_turnover>5e8) 的比例
5) Console 表格即可
"""

import argparse
import math
import os
import sys
from statistics import median

# allow running as: python tools/analyze_momentum_distribution.py ...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import db_conn


def percentile(sorted_vals, p: float):
    if not sorted_vals:
        return None
    # nearest-rank method
    k = int(math.ceil(p * len(sorted_vals))) - 1
    k = max(0, min(len(sorted_vals) - 1, k))
    return sorted_vals[k]


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="最近 N 個交易日（預設 60）")
    args = ap.parse_args()

    n_days = int(args.days)
    if n_days <= 0:
        raise ValueError("--days must be > 0")

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT trading_date
                FROM stock_signals_v2
                ORDER BY trading_date DESC
                LIMIT %s
                """,
                (n_days,),
            )
            ds = [r["trading_date"] for r in cur.fetchall()]
            if not ds:
                print("[WARN] stock_signals_v2 has no data")
                return

            placeholders = ",".join(["%s"] * len(ds))
            cur.execute(
                f"""
                SELECT momentum_score, yesterday_turnover
                FROM stock_signals_v2
                WHERE trading_date IN ({placeholders})
                """,
                tuple(ds),
            )
            rows = cur.fetchall()

    ms_all = []
    cnt_total = 0
    cnt_ms = 0
    cnt_both = 0

    for r in rows or []:
        cnt_total += 1
        ms = r.get("momentum_score")
        yto = r.get("yesterday_turnover")
        try:
            ms_f = float(ms) if ms is not None else None
        except Exception:
            ms_f = None
        try:
            yto_i = int(yto) if yto is not None else 0
        except Exception:
            yto_i = 0

        if ms_f is not None:
            ms_all.append(ms_f)
        if ms_f is not None and ms_f > 7.5:
            cnt_ms += 1
            if yto_i > 500_000_000:
                cnt_both += 1

    ms_sorted = sorted(ms_all)
    stats = {
        "min": ms_sorted[0] if ms_sorted else None,
        "median": median(ms_sorted) if ms_sorted else None,
        "p90": percentile(ms_sorted, 0.90),
        "p95": percentile(ms_sorted, 0.95),
        "max": ms_sorted[-1] if ms_sorted else None,
    }

    ratio_ms = (cnt_ms / cnt_total) if cnt_total > 0 else 0.0
    ratio_both = (cnt_both / cnt_total) if cnt_total > 0 else 0.0

    print("=== momentum_score distribution (stock_signals_v2) ===")
    print(f"trading_days: {len(ds)} (requested {n_days})")
    print(f"rows: {cnt_total}, rows_with_momentum_score: {len(ms_all)}")
    print("")
    print("| metric | value |")
    print("|---|---:|")
    print(f"| min | {fmt(stats['min'])} |")
    print(f"| median | {fmt(stats['median'])} |")
    print(f"| p90 | {fmt(stats['p90'])} |")
    print(f"| p95 | {fmt(stats['p95'])} |")
    print(f"| max | {fmt(stats['max'])} |")
    print("")
    print("| condition | ratio | count |")
    print("|---|---:|---:|")
    print(f"| momentum_score > 7.5 | {ratio_ms:.4%} | {cnt_ms} |")
    print(f"| momentum_score > 7.5 AND yesterday_turnover > 5e8 | {ratio_both:.4%} | {cnt_both} |")


if __name__ == "__main__":
    main()

