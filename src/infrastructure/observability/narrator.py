"""Turn titles the model writes for what the pipeline does, in order, off the hot path."""

import asyncio
import logging
import re  # NEW: import re
from typing import Any

from src.core.config import settings  # NEW: import settings
from src.infrastructure.observability.turns import TurnStatus, TurnType, turn_buffer
from src.infrastructure.openrouter import LLMError, OpenRouterClient
from src.infrastructure.prompt.loader import load

logger = logging.getLogger("rag.narrator")

_TITLE_TIMEOUT = 4.0
_DRAIN_TIMEOUT = 6.0
_MAX_LEN = 100

# NEW: Regex to drop titles containing URLs or HTML tags
_BAD_TITLE = re.compile(r"https?:|www\.|[\[\]|<>]")


class Narrator:
    """`say` queues a step; titles are written in the background and emitted strictly in call order.

    The model gets the step's facts and the user's question, so the same step reads differently per
    question. When the model is missing or slow, the plain `fallback` text is emitted instead.
    """

    def __init__(self, run: str, lang: str, question: str, llm: OpenRouterClient | None) -> None:
        self._run, self._lang, self._question, self._llm = run, lang, question, llm
        self._tail: asyncio.Task[None] | None = None

    def say(
        self,
        step: str,
        facts: str,
        fallback: str,
        *,
        type: TurnType = "processing",
        status: TurnStatus = "in_progress",
    ) -> None:
        self._tail = asyncio.create_task(
            self._emit(self._tail, step, facts, fallback, type, status)
        )

    async def _emit(
        self,
        prev: asyncio.Task[None] | None,
        step: str,
        facts: str,
        fallback: str,
        type: TurnType,
        status: TurnStatus,
    ) -> None:
        if prev:
            await asyncio.wait({prev})
        title = await self._title(step, facts) or fallback
        turn_buffer.emit(self._run, title, type=type, status=status)

    async def _title(self, step: str, facts: str) -> str | None:
        # NEW: Added setting toggle check
        if self._llm is None or not settings.rag_llm_titles:
            return None

        user = f"Langkah: {step}\nFAKTA: {facts}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": load(f"turn_title_system_{self._lang}")},
            {"role": "user", "content": user},
        ]

        try:
            out = await self._llm.complete(
                messages,
                json_mode=False,
                max_tokens=40,
                timeout_seconds=_TITLE_TIMEOUT,
                purpose=f"turn_title:{step}",
            )
        except LLMError:
            return None

        line = " ".join(out.split()).strip("\"'`")

        # NEW: Fallback if title is empty or contains bad patterns (URL/HTML)
        if not line or _BAD_TITLE.search(line):
            return None

        # NEW: Fallback if title contains numbers that don't exist in the facts
        if any(n not in facts for n in re.findall(r"\d+", line)):
            return None

        return line[:_MAX_LEN]

    async def drain(self) -> None:
        """Wait for queued titles so the response has the full trail; give up after a while."""
        if self._tail is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._tail), _DRAIN_TIMEOUT)
        except TimeoutError:
            logger.warning("turn titles still pending; returning without them")
