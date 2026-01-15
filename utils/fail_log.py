from __future__ import annotations

import traceback
from datetime import datetime
from typing import Optional

from utils.db import db_conn


def log_fail(
    *,
    table_name: str,
    stock_id: str,
    stage: str,
    error: Exception,
    trading_date: Optional[str] = None,
) -> None:
    """
    寫入 fail_log，供「可續跑/可觀測」使用。
    - trading_date 可空（例如全市場跑到某檔就炸）
    """
    err_type = type(error).__name__
    err_msg = str(error)
    tb = traceback.format_exc()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with db_conn(commit_on_success=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fail_log (table_name, stock_id, stage, trading_date, error_type, error_message, traceback, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (table_name, stock_id, stage, trading_date, err_type, err_msg, tb, now),
            )

