"""Run identity, stage timing and the run log.

RunRecorder gives a run an id, times each stage, and (inside `logging_to`) writes everything the
pipeline logs to a file with the run id on every line. That makes run.log the trail that a report
can be checked against: the same run id, one "started" and "finished" line per stage.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from pydantic import BaseModel, Field

log = logging.getLogger("pipeline.run")
PACKAGE_LOGGER = "pipeline"


class StageTiming(BaseModel):
    name: str = Field(min_length=1)
    duration_s: float = Field(ge=0)
    items: int | None = Field(default=None, ge=0)  # what the stage processed, when that is a count


class LogCounts(BaseModel):
    warnings: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)


class StageNote:
    """Handed to the code inside a stage so it can say how many items it processed."""

    def __init__(self) -> None:
        self.items: int | None = None


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


class _CountingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts = LogCounts()

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.counts.errors += 1
        else:
            self.counts.warnings += 1


class RunRecorder:
    def __init__(
        self,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        clock: Callable[[], float] = time.perf_counter,
        run_id: str | None = None,
    ) -> None:
        self._now, self._clock = now, clock
        self.started_at = now()
        self._t0 = clock()
        self.finished_at: datetime | None = None
        self.run_id = run_id or f"{self.started_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        self.stages: list[StageTiming] = []
        self.log_path: Path | None = None
        self._counter: _CountingHandler | None = None
        self._file_handler: logging.FileHandler | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[StageNote]:
        """Time a stage and log its start and end. The end line is logged even if the stage fails."""
        note = StageNote()
        log.info("stage %s started", name)
        started = self._clock()
        try:
            yield note
        finally:
            timing = StageTiming(name=name, duration_s=round(self._clock() - started, 4), items=note.items)
            self.stages.append(timing)
            log.info("stage %s finished in %.3fs%s", name, timing.duration_s,
                     "" if note.items is None else f" items={note.items}")

    @contextmanager
    def logging_to(self, path: Path | None, level: str = "INFO") -> Iterator[None]:
        """Write the pipeline's logs to `path` (appending) with the run id on every line, and
        count warnings and errors. Without a path only the counting happens."""
        package = logging.getLogger(PACKAGE_LOGGER)
        old_level = package.level
        run_filter = _RunIdFilter(self.run_id)
        self._counter = _CountingHandler()
        package.addHandler(self._counter)
        handler = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            handler.addFilter(run_filter)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s run=%(run_id)s %(name)s: %(message)s"
            ))
            package.addHandler(handler)
            package.setLevel(handler.level)
            self.log_path, self._file_handler = path, handler
        try:
            yield
        finally:
            if handler is not None:
                handler.flush()
                handler.close()
                package.removeHandler(handler)
            package.removeHandler(self._counter)
            package.setLevel(old_level)

    def flush_log(self) -> None:
        if self._file_handler is not None:
            self._file_handler.flush()

    def finish(self) -> None:
        self.finished_at = self._now()

    @property
    def duration_s(self) -> float:
        return round(self._clock() - self._t0, 4)

    @property
    def log_counts(self) -> LogCounts:
        return self._counter.counts if self._counter else LogCounts()
