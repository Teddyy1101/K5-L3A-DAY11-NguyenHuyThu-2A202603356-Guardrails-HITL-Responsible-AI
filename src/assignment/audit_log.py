"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from agents.security_boundary import normalize_for_security
from assignment._artifacts import write_json_artifact
from guardrails.output_guardrails import content_filter


def _preview(text: str) -> str:
    return content_filter(normalize_for_security(text))["redacted"][:300]


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[str, float, str]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Remember the sanitized input and start time for a request."""
        self._open[(user_id, request_id)] = (utc_now_iso(), time.monotonic(), _preview(text))

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Append the sanitized response and the decision to the audit log."""
        key = (user_id, request_id)
        pending = self._open.get(key)
        if pending is None:
            timestamp, input_preview, latency_ms = utc_now_iso(), None, None
        else:
            timestamp, started_at, input_preview = pending
            latency_ms = round((time.monotonic() - started_at) * 1000, 3)

        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "input": input_preview,
                "output": _preview(text),
                "blocked": blocked,
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )
        self._open.pop(key, None)

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        write_json_artifact(filepath or default_audit_log_path(), self.logs)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
