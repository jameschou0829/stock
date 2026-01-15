from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_env_value(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    # 嘗試 JSON（數字/布林/陣列/物件）
    if text[0] in "{[\"'" or text.isdigit() or text.lower() in ("true", "false", "null"):
        try:
            return json.loads(text)
        except Exception:
            pass
    # 嘗試數字
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def _env_overrides(prefix: str = "STRATEGY__") -> Dict[str, Any]:
    """
    允許用環境變數覆蓋策略設定：
      STRATEGY__SCORING__ENTRY_MIN_SCORE_LONG=30
    """
    out: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        path = k[len(prefix):].lower().split("__")
        cur = out
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
        cur[path[-1]] = _parse_env_value(v)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"strategy config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("strategy.yaml must be a mapping")
    return data


def load_strategy_config(overrides: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str, str]:
    """
    回傳 (config, config_hash, config_snapshot)
    覆蓋優先順序：
      1) overrides（CLI / web request）
      2) 環境變數 STRATEGY__*
      3) configs/strategy.yaml
    """
    root = Path(__file__).resolve().parents[1]
    base = _load_yaml(root / "configs" / "strategy.yaml")
    _deep_update(base, _env_overrides())
    if overrides:
        _deep_update(base, overrides)
    snapshot = json.dumps(base, ensure_ascii=False, sort_keys=True)
    config_hash = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    return base, config_hash, snapshot
