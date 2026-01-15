from __future__ import annotations

from utils.db import db_conn


def has_column(table_name: str, column_name: str) -> bool:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = %s
                  AND COLUMN_NAME = %s
                LIMIT 1
                """,
                (table_name, column_name),
            )
            return cur.fetchone() is not None
