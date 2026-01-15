# stock_list.py

from utils.db import db_conn

def get_active_stock_list():
    """
    從 stock_info 撈出 is_active = 1 的股票代碼
    """
    with db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT stock_id
                FROM stock_info
                WHERE is_active = 1
                ORDER BY stock_id
                """
            )
            rows = cursor.fetchall()
            return [r["stock_id"] for r in rows]
