from datetime import date, timedelta, datetime
from utils.db import db_conn
from utils.fetcher import FinMindPaymentRequiredError, FinMindAuthError, finmind_get_data
from typing import Optional
from utils.fetch_log import get_last_date as _get_last_date, upsert_fetch_log as _upsert_fetch_log

DATASET = "TaiwanStockMarginPurchaseShortSale"
DEFAULT_START = date(2015, 1, 1)
TABLE_NAME = "margin_trading"

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


def fetch_margin(stock_id, start_date):
    params = {
        "data_id": stock_id,  # ⚠️ FinMind 使用 data_id
        "start_date": start_date.strftime("%Y-%m-%d"),
    }
    return finmind_get_data(DATASET, params, timeout=30, max_retry=3, wait_seconds=2)


def save_margin(conn, stock_id, rows):
    if not rows:
        return

    sql = """
    INSERT INTO margin_trading (
        stock_id,
        trading_date,
        margin_buy,
        margin_sell,
        margin_balance,
        short_buy,
        short_sell,
        short_balance
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        margin_buy = VALUES(margin_buy),
        margin_sell = VALUES(margin_sell),
        margin_balance = VALUES(margin_balance),
        short_buy = VALUES(short_buy),
        short_sell = VALUES(short_sell),
        short_balance = VALUES(short_balance)
    """

    values = []
    for r in rows:
        d = _parse_finmind_date(r.get("date"))
        if d is None:
            continue
        values.append((
            stock_id,
            d,
            r.get("MarginPurchaseBuy", 0),
            r.get("MarginPurchaseSell", 0),
            r.get("MarginPurchaseBalance", 0),
            r.get("ShortSaleBuy", 0),
            r.get("ShortSaleSell", 0),
            r.get("ShortSaleBalance", 0),
        ))
    if not values:
        return

    with conn.cursor() as cur:
        cur.executemany(sql, values)


def run_margin(stock_id, end_date: Optional[date] = None):
    """
    end_date: 上層可指定「最新可用交易日」，避免今天資料未出造成不必要的更新/判斷。
    FinMind 此 dataset 無 end_date 參數，所以拉回來後會把 >end_date 的資料濾掉。
    """
    if end_date is None:
        end_date = date.today()
    try:
        # checkpoint：以 fetch_log 為主
        last_date = _get_last_date(TABLE_NAME, stock_id)

        if last_date is None:
            start_date = DEFAULT_START
            print(f"[INIT] 融資融券補歷史 {stock_id} from {start_date}")
        else:
            start_date = last_date + timedelta(days=1)
            if start_date > end_date:
                print(f"[SKIP] 融資無新資料 {stock_id}")
                return
            print(f"[UPDATE] 融資融券補資料 {stock_id} from {start_date}")

        rows = fetch_margin(stock_id, start_date)
        if not rows:
            print(f"[WARN] 融資 API 無資料 {stock_id}")
            return

        # 避免把 end_date 之後的資料寫進 DB
        rows2 = []
        for r in rows:
            d = _parse_finmind_date(r.get("date"))
            if d is None:
                continue
            if d <= end_date:
                rows2.append(r)
        rows = rows2
        if not rows:
            print(f"[SKIP] 融資：無 end_date({end_date}) 前可寫入資料 {stock_id}")
            return

        with db_conn(commit_on_success=True) as conn:
            save_margin(conn, stock_id, rows)

        # 更新 checkpoint（取本次寫入最大日期）
        max_d = None
        for r in rows:
            dd = _parse_finmind_date(r.get("date"))
            if dd is None:
                continue
            if max_d is None or dd > max_d:
                max_d = dd
        if max_d is not None:
            _upsert_fetch_log(TABLE_NAME, stock_id, max_d)

        print(f"[OK] 融資完成 {stock_id} ({len(rows)} 筆)")

    except (FinMindPaymentRequiredError, FinMindAuthError):
        raise
    except Exception as e:
        print(f"[ERROR] 融資 ETL 失敗 {stock_id}: {e}")
