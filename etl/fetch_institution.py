from datetime import date, timedelta, datetime
from collections import defaultdict
from typing import Optional

from utils.db import db_conn
from utils.fetcher import FinMindPaymentRequiredError, FinMindAuthError, finmind_get_data

TABLE_NAME = "institution_trading"
DEFAULT_START_DATE = date(2015, 1, 1)

def _parse_finmind_date(v):
    """
    FinMind 的 date 欄位有時是 'YYYY-MM-DD' 字串，有時可能已是 datetime.date。
    統一轉成 datetime.date，避免與 end_date 比較時 TypeError。
    """
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        s = str(v)
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


# -----------------------------
# DB helpers
# -----------------------------
def get_last_date(conn, stock_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT last_date
            FROM fetch_log
            WHERE table_name=%s AND stock_id=%s
            """,
            (TABLE_NAME, stock_id),
        )
        row = cur.fetchone()
        return row["last_date"] if row else None


def update_fetch_log(conn, stock_id, last_date):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fetch_log (table_name, stock_id, last_date)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE last_date=VALUES(last_date)
            """,
            (TABLE_NAME, stock_id, last_date),
        )


# -----------------------------
# API fetch
# -----------------------------
def fetch_institution(stock_id, start_date):
    params = {
        "data_id": stock_id,  # ⚠️ FinMind 使用 data_id
        "start_date": start_date.strftime("%Y-%m-%d"),
    }
    return finmind_get_data("TaiwanStockInstitutionalInvestorsBuySell", params, timeout=30, max_retry=3, wait_seconds=2)


# -----------------------------
# normalize (長表 → 日彙總)
# -----------------------------
def normalize_rows(rows):
    daily = defaultdict(lambda: {
        "foreign_buy": 0,
        "foreign_sell": 0,
        "trust_buy": 0,
        "trust_sell": 0,
        "dealer_buy": 0,
        "dealer_sell": 0,
    })

    for r in rows:
        d = _parse_finmind_date(r.get("date"))
        if d is None:
            continue
        name = r["name"]
        buy = int(r.get("buy", 0))
        sell = int(r.get("sell", 0))

        if name == "Foreign_Investor":
            daily[d]["foreign_buy"] += buy
            daily[d]["foreign_sell"] += sell

        elif name == "Investment_Trust":
            daily[d]["trust_buy"] += buy
            daily[d]["trust_sell"] += sell

        elif name in ("Dealer_self", "Dealer_Hedging"):
            daily[d]["dealer_buy"] += buy
            daily[d]["dealer_sell"] += sell

    return daily


# -----------------------------
# save
# -----------------------------
def save_institution(conn, stock_id, daily):
    with conn.cursor() as cur:
        for d, v in daily.items():
            cur.execute(
                """
                INSERT INTO institution_trading
                (stock_id, trading_date,
                 foreign_buy, foreign_sell, foreign_net,
                 trust_buy, trust_sell, trust_net,
                 dealer_buy, dealer_sell, dealer_net)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                 foreign_buy=VALUES(foreign_buy),
                 foreign_sell=VALUES(foreign_sell),
                 foreign_net=VALUES(foreign_net),
                 trust_buy=VALUES(trust_buy),
                 trust_sell=VALUES(trust_sell),
                 trust_net=VALUES(trust_net),
                 dealer_buy=VALUES(dealer_buy),
                 dealer_sell=VALUES(dealer_sell),
                 dealer_net=VALUES(dealer_net)
                """,
                (
                    stock_id,
                    d,
                    v["foreign_buy"],
                    v["foreign_sell"],
                    v["foreign_buy"] - v["foreign_sell"],
                    v["trust_buy"],
                    v["trust_sell"],
                    v["trust_buy"] - v["trust_sell"],
                    v["dealer_buy"],
                    v["dealer_sell"],
                    v["dealer_buy"] - v["dealer_sell"],
                ),
            )


# -----------------------------
# main runner
# -----------------------------
def run_institution(stock_id, end_date: Optional[date] = None):
    """
    end_date: 上層可指定「最新可用交易日」，避免今天資料未出時每檔都白打 API。
    注意：FinMind 這支 dataset 沒有 end_date 參數，所以我們用：
    - last_date >= end_date -> 直接 skip
    - 拉回來後再把 > end_date 的日期濾掉，避免寫入不存在的「未完整日」
    """
    if end_date is None:
        end_date = date.today()
    try:
        with db_conn(commit_on_success=True) as conn:
            last_date = get_last_date(conn, stock_id)

            if last_date:
                start_date = last_date + timedelta(days=1)
                if start_date > end_date:
                    print(f"[SKIP] 法人無新資料 {stock_id}")
                    return
            else:
                start_date = DEFAULT_START_DATE
                print(f"[INIT] 法人補歷史 {stock_id} from {start_date}")

            rows = fetch_institution(stock_id, start_date)
            if not rows:
                print(f"[WARN] 法人 API 無資料 {stock_id}")
                return

            daily = normalize_rows(rows)
            # 避免把 end_date 之後的資料寫進 DB（例如今天資料尚未完整）
            daily = {d: v for d, v in daily.items() if (d is not None and d <= end_date)}
            if not daily:
                print(f"[SKIP] 法人：無 end_date({end_date}) 前可寫入資料 {stock_id}")
                return
            save_institution(conn, stock_id, daily)

            max_date = max(daily.keys())
            update_fetch_log(conn, stock_id, max_date)

            print(f"[OK] 法人完成 {stock_id} ({len(daily)} 天)")

    except (FinMindPaymentRequiredError, FinMindAuthError):
        # 讓上層可以 fail fast 中止整批
        raise
    except Exception as e:
        print(f"[ERROR] 法人 ETL 失敗 {stock_id}: {e}")
