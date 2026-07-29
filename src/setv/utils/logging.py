"""Structured JSONL event logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    """Append-only JSONL logger for preflight events."""

    def __init__(self, path: str | Path, *, echo: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        if self.echo:
            print(payload, flush=True)
