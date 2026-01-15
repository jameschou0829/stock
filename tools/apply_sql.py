import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql

from utils.db import db_conn


def _strip_sql_comments(sql: str) -> str:
    out = []
    for line in sql.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("--"):
            continue
        out.append(line)
    return "\n".join(out)


def _split_statements(sql: str) -> list[str]:
    # naive split by ';' (migrations in this repo are simple DDL/ALTER/CREATE INDEX)
    parts = []
    buf = []
    for ch in sql:
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                parts.append(stmt)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def apply_sql_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    raw = p.read_text(encoding="utf-8")
    cleaned = _strip_sql_comments(raw)
    stmts = _split_statements(cleaned)
    if not stmts:
        print(f"[apply_sql] no statements: {path}")
        return

    with db_conn(commit_on_success=True) as conn:
        with conn.cursor() as cur:
            for i, stmt in enumerate(stmts, 1):
                try:
                    cur.execute(stmt)
                    print(f"[apply_sql] OK {i}/{len(stmts)}")
                except pymysql.err.OperationalError as e:
                    # Allow re-run migrations:
                    # 1060: Duplicate column name
                    # 1061: Duplicate key name
                    errno = e.args[0] if e.args else None
                    if errno in (1060, 1061):
                        print(f"[apply_sql] SKIP duplicate (errno={errno}) {i}/{len(stmts)}")
                        continue
                    raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/apply_sql.py migrations/xxx.sql")
        raise SystemExit(2)
    apply_sql_file(sys.argv[1])

