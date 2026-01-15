# run_init_etl.py
from etl.fetch_daily import run_daily
from utils.db import db_conn

def get_stock_list():
    with db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT stock_id
                FROM stock_info
                WHERE is_active = 1
                  AND stock_id REGEXP '^[0-9]+$'
                """
            )
            return [r["stock_id"] for r in cursor.fetchall()]

def run_init():
    stock_list = get_stock_list()
    print(f"初始化股票數：{len(stock_list)}")

    for sid in stock_list:
        try:
            run_daily(sid)
        except Exception as e:
            print(f"[ERROR] {sid} 日K 失敗: {e}")

if __name__ == "__main__":
    run_init()
