"""
utils/fetcher.py

Phase B：
- 統一 FinMind client 入口（Session + retry）
- 429/5xx exponential backoff
- 401/403/402 fail fast
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

import requests

from configs.settings import FINMIND_API_URL
from configs import settings

class FinMindPaymentRequiredError(RuntimeError):
    pass


class FinMindAuthError(RuntimeError):
    pass


class _RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = float(min_interval_sec)
        self._last_ts = 0.0

    def wait(self):
        now = time.time()
        wait = (self._last_ts + self.min_interval_sec) - now
        if wait > 0:
            time.sleep(wait)
        self._last_ts = time.time()


_rate_limiter = _RateLimiter(getattr(settings, "FINMIND_MIN_INTERVAL_SEC", 0.0))
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "stock_test3/etl"})
        _session = s
    return _session


def _raise_if_auth_or_payment(resp: requests.Response, dataset: str):
    if resp.status_code == 402:
        raise FinMindPaymentRequiredError(
            f"FinMind 回傳 402 Payment Required（dataset={dataset}）："
            f"通常是 token 方案/額度/權限問題（到期或需付費）。"
            f"請更新 FINMIND_API_TOKEN 或升級方案後再跑。"
        )
    if resp.status_code in (401, 403):
        raise FinMindAuthError(
            f"FinMind 回傳 {resp.status_code}（dataset={dataset}）：token 無效或無權限。"
            f"請更新 FINMIND_API_TOKEN 後再跑。"
        )


def finmind_get_data(
    dataset: str,
    params: dict,
    timeout: int = 30,
    max_retry: int = 5,
    wait_seconds: float = 0.5,
):
    """
    統一的 FinMind 呼叫入口（含節流與 401/402/403 快速失敗、429/5xx 退避重試）。
    回傳 list data（永遠是 list）。
    """
    token = settings.require_finmind_token()
    base_params = {"dataset": dataset, "token": token}
    base_params.update(params)

    for i in range(max_retry):
        _rate_limiter.wait()
        sess = _get_session()
        resp = sess.get(FINMIND_API_URL, params=base_params, timeout=timeout)
        _raise_if_auth_or_payment(resp, dataset)

        # 429 / 5xx：exponential backoff with jitter
        if resp.status_code == 429 or (500 <= resp.status_code <= 599):
            backoff = wait_seconds * (2 ** i) + random.random() * 0.2
            print(f"[WARN] FinMind HTTP {resp.status_code} dataset={dataset} retry {i+1}/{max_retry} sleep={backoff:.2f}s")
            time.sleep(backoff)
            continue

        resp.raise_for_status()
        j: Dict[str, Any] = resp.json()
        if j.get("status") == 200:
            return j.get("data", []) or []

        # 非 200：也做退避（有些會回 400 但仍可重試）
        backoff = wait_seconds * (2 ** i) + random.random() * 0.2
        print(f"[WARN] API status {j.get('status')}, msg={j.get('msg')}, retry {i+1}/{max_retry} sleep={backoff:.2f}s")
        time.sleep(backoff)

    return []


def fetch_with_retry(dataset: str, params: dict, max_retry: int = 5, wait_seconds: float = 0.5) -> dict:
    """
    舊介面相容：回傳 {"data": [...]}。
    新實作直接復用 finmind_get_data。
    """
    data = finmind_get_data(dataset, params, timeout=15, max_retry=max_retry, wait_seconds=wait_seconds)
    return {"data": data}
