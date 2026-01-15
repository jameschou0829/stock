"""
utils/db.py

DB 連線統一入口：使用 contextmanager，避免全域共享連線造成：
- 連線被 MySQL idle timeout 斷線後隨機炸裂
- 多執行緒/多請求共用同一連線導致游標互相干擾
- 呼叫端忘記 close 造成資源洩漏
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import pymysql

from configs.settings import DB_CHARSET, DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER


@contextmanager
def db_conn(*, commit_on_success: bool = False) -> Iterator[pymysql.connections.Connection]:
    """
    用法：
      with db_conn() as conn:
          with conn.cursor() as cur:
              cur.execute(...)

    - commit_on_success=False：預設適合查詢/讀取（避免無意義 commit）
    - commit_on_success=True：適合 ETL 寫入（若無 exception 自動 commit；否則 rollback）
    """
    conn: Optional[pymysql.connections.Connection] = None
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            charset=DB_CHARSET,
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
        yield conn
        if commit_on_success:
            conn.commit()
    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
