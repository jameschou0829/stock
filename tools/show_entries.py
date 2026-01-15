"""
tools/show_entries.py

用途：用最簡單的方式把某一天「可進場」(entry_long/entry_short) 的股票列出來，
讓你不用手寫 SQL。

用法：
  python tools/show_entries.py 2026-01-13 --side long
  python tools/show_entries.py 2026-01-13 --side short
  python tools/show_entries.py 2026-01-01 2026-01-13 --side long   # 區間

參數：
  --only-entry: 只列 entry=1（預設 True）
  --limit: 每天最多列出幾檔（預設 50）
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import db_conn


def parse_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", help="YYYY-MM-DD")
    ap.add_argument("end", nargs="?", help="YYYY-MM-DD (optional)")
    ap.add_argument("--side", default="long", choices=["long", "short"])
    ap.add_argument("--only-entry", action="store_true", help="只列出 entry=1（預設：開啟）")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else start

    entry_col = "entry_long" if args.side == "long" else "entry_short"
    score_order = "DESC" if args.side == "long" else "ASC"

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT trading_date
                FROM stock_signals_v2
                WHERE trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (start, end),
            )
            days = [r["trading_date"] for r in cur.fetchall()]

            for d in days:
                where = f"trading_date=%s"
                params = [d]
                if args.only_entry or True:
                    where += f" AND {entry_col}=1"
                cur.execute(
                    f"""
                    SELECT stock_id, market_regime, score, primary_strategy,
                           stop_loss_side, stop_loss_price, stop_loss_pct,
                           strat_volume_momentum, strat_price_volume_new_high, strat_breakout_edge,
                           strat_trust_breakout, strat_trust_momentum_buy, strat_foreign_big_buy, strat_co_buy
                    FROM stock_signals_v2
                    WHERE {where}
                    ORDER BY score {score_order}
                    LIMIT %s
                    """,
                    (*params, int(args.limit)),
                )
                rows = cur.fetchall()

                print(f"\n=== {d} | side={args.side} | {entry_col}=1 | count={len(rows)} ===")
                for r in rows:
                    tags = []
                    for k in (
                        "strat_volume_momentum",
                        "strat_price_volume_new_high",
                        "strat_breakout_edge",
                        "strat_trust_breakout",
                        "strat_trust_momentum_buy",
                        "strat_foreign_big_buy",
                        "strat_co_buy",
                    ):
                        if int(r.get(k, 0) or 0) == 1:
                            tags.append(k.replace("strat_", ""))
                    sl = r.get("stop_loss_price")
                    sl_str = f"SL={sl:.2f}" if isinstance(sl, (int, float)) and sl is not None else "SL=-"
                    print(
                        f"{r['stock_id']} score={int(r.get('score', 0) or 0)} "
                        f"regime={r.get('market_regime')} "
                        f"primary={r.get('primary_strategy')} "
                        f"{sl_str} tags={','.join(tags) or '-'}"
                    )


if __name__ == "__main__":
    main()

