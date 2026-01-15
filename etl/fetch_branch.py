from datetime import date, timedelta

from utils.db import db_conn
from utils.fetcher import FinMindPaymentRequiredError, FinMindAuthError
from collections import defaultdict
from utils.fetcher import finmind_get_data
from configs import settings
from utils.fetch_log import get_last_date as _get_last_date, upsert_fetch_log as _upsert_fetch_log

TABLE_NAME = "branch_trading"
# 注意：TaiwanStockTradingDailyReport 這個 dataset 的 start_date 行為偏「指定單日」，
# 若 start_date 不是交易日，常會回傳空資料。
# 因此這裡改成：從 stock_daily 取最近 N 個「交易日」逐日抓取。
LOOKBACK_TRADING_DAYS = 5


def fetch_branch(stock_id, start_date):
    params = {
        "data_id": stock_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
    }
    # 注意：若 start_date 不是交易日，FinMind 可能回空 list（不視為錯誤）
    return finmind_get_data("TaiwanStockTradingDailyReport", params, timeout=30, max_retry=2, wait_seconds=1)


def get_recent_trading_dates(stock_id: str, limit: int):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date
                FROM stock_daily
                WHERE stock_id=%s
                ORDER BY trading_date DESC
                LIMIT %s
                """,
                (stock_id, int(limit)),
            )
            rows = cur.fetchall()
            return [r["trading_date"] for r in rows]


def run_branch(stock_id):
    # checkpoint：以 fetch_log 為主
    last_branch_date = _get_last_date(TABLE_NAME, stock_id)

    trading_dates = get_recent_trading_dates(stock_id, LOOKBACK_TRADING_DAYS)

    # 避免「今天資料不完整」造成白打 API：只抓到最新完整交易日為止
    latest_complete_date = None
    min_stocks = getattr(settings, "SIGNALS_MIN_STOCKS", 2000)
    with db_conn() as conn1:
        with conn1.cursor() as cur1:
            cur1.execute(
                """
                SELECT trading_date
                FROM stock_daily
                GROUP BY trading_date
                HAVING COUNT(DISTINCT stock_id) >= %s
                ORDER BY trading_date DESC
                LIMIT 1
                """,
                (int(min_stocks),),
            )
            row = cur1.fetchone()
            latest_complete_date = row["trading_date"] if row else None

    if latest_complete_date:
        trading_dates = [d for d in trading_dates if d <= latest_complete_date]
    if last_branch_date:
        trading_dates = [d for d in trading_dates if d > last_branch_date]
    if not trading_dates:
        print(f"[SKIP] 分點：無需更新 {stock_id}")
        return

    with db_conn(commit_on_success=True) as conn:
        try:
            total_raw_rows = 0
            total_agg_rows = 0
            with conn.cursor() as cur:
                for d in trading_dates:
                    rows = fetch_branch(stock_id, d)
                    if rows is None:
                        print(f"[SKIP] 分點無資料 {stock_id} {d} (422)")
                        continue
                    if not rows:
                        print(f"[SKIP] 分點無資料 {stock_id} {d}")
                        continue

                    # TaiwanStockTradingDailyReport 具有 price 維度，若直接寫入會被 UNIQUE KEY 覆寫。
                    # 這裡先彙總成「券商/分點（日）總買賣」後再寫入，符合 branch_trading schema。
                    agg = defaultdict(lambda: {"buy": 0, "sell": 0})
                    for r in rows:
                        key = (
                            r["date"],
                            r.get("securities_trader", ""),
                            r.get("securities_trader_id", ""),
                        )
                        agg[key]["buy"] += int(r.get("buy", 0) or 0)
                        agg[key]["sell"] += int(r.get("sell", 0) or 0)

                    for (trading_date, broker, branch), v in agg.items():
                        cur.execute(
                            """
                            INSERT INTO branch_trading
                            (stock_id, trading_date, broker, branch, buy, sell, net)
                            VALUES (%s,%s,%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE
                            buy=VALUES(buy),
                            sell=VALUES(sell),
                            net=VALUES(net)
                            """,
                            (
                                stock_id,
                                trading_date,
                                broker,
                                branch,
                                v["buy"],
                                v["sell"],
                                int(v["buy"]) - int(v["sell"]),
                            ),
                        )
                    total_raw_rows += len(rows)
                    total_agg_rows += len(agg)

            print(
                f"[OK] 分點完成 {stock_id} "
                f"(raw={total_raw_rows} 筆, agg={total_agg_rows} 筆 / {len(trading_dates)} 交易日)"
            )

            # 更新 checkpoint：成功處理的最大交易日
            if trading_dates:
                _upsert_fetch_log(TABLE_NAME, stock_id, max(trading_dates))

        except (FinMindPaymentRequiredError, FinMindAuthError):
            raise
        except Exception as e:
            print(f"[ERROR] 分點 ETL 失敗 {stock_id}: {e}")
