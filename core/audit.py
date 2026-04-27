"""
Append-only JSONL audit log for every PBPK simulation.

Each call to a registered tool writes one JSON line containing the
inputs, the resolved (post-validation) parameters, the soft warnings
emitted, the result fingerprint, and a wall-clock timestamp.

Replay is supported by hashing the input record and storing it as
`fingerprint`. Same input → same fingerprint → reproducible run.
"""

from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


_AUDIT_DIR = Path(__file__).resolve().parent.parent / "data"
_AUDIT_LOG = _AUDIT_DIR / "audit.jsonl"


def _stable_hash(obj: Any) -> str:
    """Deterministic hash of arbitrary JSON-serializable payload."""
    try:
        s = json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        s = repr(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def log_simulation(
    *,
    tool_name: str,
    inputs: dict,
    resolved: Optional[dict] = None,
    warnings: Optional[list[str]] = None,
    summary: Optional[dict] = None,
    error: Optional[str] = None,
) -> str:
    """
    Append a record to data/audit.jsonl. Returns the fingerprint.

    The fingerprint is computed from `inputs` only (not `resolved`),
    so two calls with the same user-supplied parameters share an ID.
    """
    record = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool_name,
        "inputs": inputs,
        "resolved": resolved or {},
        "warnings": warnings or [],
        "summary": summary or {},
        "error": error,
    }
    fp = _stable_hash({"tool": tool_name, "inputs": inputs})
    record["fingerprint"] = fp
    try:
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        # Audit failure must not break a simulation — log to stderr instead
        print(f"[audit] failed to write log: {e}", file=sys.stderr)
    return fp


def replay_lookup(fingerprint: str) -> Optional[dict]:
    """Find the most recent record with this fingerprint."""
    if not _AUDIT_LOG.exists():
        return None
    matches = []
    with open(_AUDIT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("fingerprint") == fingerprint:
                matches.append(rec)
    return matches[-1] if matches else None


def audit_log_path() -> Path:
    return _AUDIT_LOG
