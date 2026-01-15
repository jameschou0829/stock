from datetime import date

from utils.db import db_conn


def default_start_date(*args, **kwargs):
    return date(2015, 1, 1)


def get_last_date(table_name: str, stock_id: str):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT last_date
                FROM fetch_log
                WHERE table_name = %s AND stock_id = %s
                """,
                (table_name, stock_id),
            )
            row = cur.fetchone()
            return row["last_date"] if row else None


def upsert_fetch_log(table_name: str, stock_id: str, last_date):
    with db_conn(commit_on_success=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fetch_log (table_name, stock_id, last_date)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    last_date = VALUES(last_date)
                """,
                (table_name, stock_id, last_date),
            )
