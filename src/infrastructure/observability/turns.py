"""Turn titles: short, user-facing step labels emitted while a chat request runs.

One in-process buffer keyed by the run id the caller sends. The backend polls it while the
request is in flight and the final response embeds the same trail, so the UI can show live
progress and keep it afterwards. One `ai` process per deployment makes a dict sufficient;
swap it for Redis if this ever scales horizontally.
"""

import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel

TurnType = Literal["processing", "result", "error"]
TurnStatus = Literal["in_progress", "done", "failed"]
_MAX_PER_RUN = 32
_TTL_SECONDS = 900


class TurnEvent(BaseModel):
    turn_id: str
    run_id: str
    sequence: int
    title: str
    type: TurnType = "processing"
    status: TurnStatus = "in_progress"
    timestamp: str


class TurnBuffer:
    def __init__(self) -> None:
        self._runs: dict[str, deque[TurnEvent]] = defaultdict(lambda: deque(maxlen=_MAX_PER_RUN))
        self._seq: dict[str, int] = defaultdict(int)
        self._touched: dict[str, float] = {}

    def emit(
        self,
        run_id: str,
        title: str,
        *,
        type: TurnType = "processing",
        status: TurnStatus = "in_progress",
    ) -> TurnEvent:
        self._seq[run_id] += 1
        ev = TurnEvent(
            turn_id=str(uuid.uuid4()),
            run_id=run_id,
            sequence=self._seq[run_id],
            title=title,
            type=type,
            status=status,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._runs[run_id].append(ev)
        self._touched[run_id] = time.monotonic()
        return ev

    def read(self, run_id: str, since: int = 0) -> list[TurnEvent]:
        self._evict()
        return [e for e in self._runs.get(run_id, ()) if e.sequence > since]

    def _evict(self) -> None:
        now = time.monotonic()
        for rid in [r for r, t in self._touched.items() if now - t > _TTL_SECONDS]:
            self._runs.pop(rid, None)
            self._seq.pop(rid, None)
            self._touched.pop(rid, None)


turn_buffer = TurnBuffer()


def dump(events: list[TurnEvent]) -> list[dict[str, Any]]:
    return [e.model_dump() for e in events]
