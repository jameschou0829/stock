from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from configs.strategy_loader import load_strategy_config
from utils.db import db_conn


def _is_safe_ident(name: str) -> bool:
    return bool(name) and all(c.isalnum() or c == "_" for c in name)


def _get_columns(conn, table: str) -> Set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
            """,
            (table,),
        )
        return {r["COLUMN_NAME"] for r in (cur.fetchall() or [])}


def _load_signals(start: date, end: date, features: List[str], prob_field: str) -> pd.DataFrame:
    # Build SELECT with graceful fallback for missing columns.
    with db_conn() as conn:
        cols = _get_columns(conn, "stock_signals_v2")
        if prob_field and not _is_safe_ident(prob_field):
            raise RuntimeError(f"invalid prob_field: {prob_field}")
        for f in features:
            if not _is_safe_ident(f):
                raise RuntimeError(f"invalid feature name: {f}")

        select_exprs = ["stock_id", "trading_date", "close"]
        for f in features:
            if f in cols:
                select_exprs.append(f)
            elif f == "score_long":
                select_exprs.append("score AS score_long")
            elif f == "score_short":
                select_exprs.append("ABS(score) AS score_short")
            else:
                # Missing feature column -> NULL (will be filled with 0.0)
                select_exprs.append(f"NULL AS {f}")

        cols_sql = ", ".join(select_exprs)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {cols_sql}
                FROM stock_signals_v2
                WHERE trading_date BETWEEN %s AND %s
                """,
                (start, end),
            )
            rows = cur.fetchall() or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


def _load_daily(start: date, end: date) -> pd.DataFrame:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stock_id, trading_date, high, close
                FROM stock_daily
                WHERE trading_date BETWEEN %s AND %s
                """,
                (start, end),
            )
            rows = cur.fetchall() or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


def _compute_labels(df: pd.DataFrame, daily: pd.DataFrame, label_cfg: dict) -> pd.Series:
    mode = label_cfg.get("mode", "kpi_touch")
    horizon = int(label_cfg.get("horizon_days", 5) or 5)
    target = float(label_cfg.get("target_return", 0.0) or 0.0)
    price_source = str(label_cfg.get("price_source", "high"))

    daily = daily.sort_values(["stock_id", "trading_date"])
    daily_grp = daily.groupby("stock_id", sort=False)

    labels: list[float] = []
    for _, row in df.iterrows():
        sid = row["stock_id"]
        d = row["trading_date"]
        entry_px = float(row.get("close") or 0.0)
        if entry_px <= 0:
            labels.append(np.nan)
            continue
        if sid not in daily_grp.indices:
            labels.append(np.nan)
            continue
        g = daily_grp.get_group(sid)
        idx = g.index[g["trading_date"] == d]
        if len(idx) == 0:
            labels.append(np.nan)
            continue
        i = g.index.get_loc(idx[0])
        future = g.iloc[i + 1 : i + 1 + horizon]
        if future.empty:
            # not enough future bars for label
            labels.append(np.nan)
            continue
        if mode == "close_positive":
            px = float(future.iloc[-1]["close"])
            labels.append(1 if px > entry_px else 0)
            continue
        # kpi_touch
        if price_source == "close":
            px = float(future["close"].max())
        else:
            px = float(future["high"].max())
        labels.append(1 if px >= entry_px * (1.0 + target) else 0)
    return pd.Series(labels, index=df.index)


def train_and_update_probabilities(
    *,
    start: date,
    end: date,
    train_end: Optional[date] = None,
    overrides: Optional[dict] = None,
) -> int:
    strategy_cfg, _, _ = load_strategy_config(overrides)
    prob_cfg = strategy_cfg.get("probability", {})
    if not prob_cfg.get("enabled", True):
        return 0
    features = prob_cfg.get("features", [])
    prob_field = prob_cfg.get("prob_field", "prob_kpi_5d_2pct_touch")
    label_cfg = prob_cfg.get("label", {})
    horizon = int(label_cfg.get("horizon_days", 5) or 5)

    if not features:
        raise RuntimeError("probability.features is empty")

    if train_end is None:
        train_end = end
    if train_end < start or train_end > end:
        raise RuntimeError(f"invalid train_end: {train_end}")

    # Ensure prob_field exists in DB
    with db_conn() as conn:
        cols = _get_columns(conn, "stock_signals_v2")
        if prob_field not in cols:
            raise RuntimeError(
                f"prob_field column not found: {prob_field}. "
                f"Please apply migrations/20260115_extend_stock_signals_v2.sql (and/or add prob column)."
            )

    df_all = _load_signals(start, end, features, prob_field)
    if df_all.empty:
        return 0

    # For training labels we need future bars up to train_end + horizon
    daily = _load_daily(start, min(end, train_end + timedelta(days=horizon * 3)))
    if daily.empty:
        return 0

    df_train = df_all[df_all["trading_date"].dt.date <= train_end].copy()
    df_pred = df_all[df_all["trading_date"].dt.date > train_end].copy()

    df_train["label"] = _compute_labels(df_train, daily, label_cfg)
    df_train = df_train.dropna(subset=["label"])
    if df_train.empty:
        raise RuntimeError("no train samples after label computation (check date range / horizon)")

    X_train = df_train[features].fillna(0.0)
    y_train = df_train["label"].astype(int)

    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # predict range: (train_end, end]
    if df_pred.empty:
        return 0
    X_pred = df_pred[features].fillna(0.0)
    probs = clf.predict_proba(X_pred)[:, 1]
    df_pred["prob"] = probs

    rows = list(zip(df_pred["prob"], df_pred["stock_id"], df_pred["trading_date"].dt.date))
    with db_conn(commit_on_success=True) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"""
                UPDATE stock_signals_v2
                SET {prob_field} = %s
                WHERE stock_id = %s AND trading_date = %s
                """,
                rows,
            )
    return len(rows)
