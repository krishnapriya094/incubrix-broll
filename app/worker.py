"""Background worker (Integration Standard point 3).

A small thread-pool queue: POST returns immediately with a job id, the pipeline
runs in the background and progress is written to the job store.
"""
from __future__ import annotations

import queue
import threading

from .core import get_logger
from . import pipeline

log = get_logger("worker")


class Worker:
    def __init__(self, store, settings, threads: int = 2):
        self.store = store
        self.settings = settings
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.threads = [
            threading.Thread(target=self._loop, name=f"worker-{i}", daemon=True)
            for i in range(max(1, threads))
        ]
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        for thread in self.threads:
            thread.start()
        self._started = True

    def submit(self, job_id: str) -> None:
        self.start()
        self.queue.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                job = self.store.get(job_id)
                if job and job["status"] in {"queued", "running"}:
                    pipeline.run_pipeline(job_id, job["request"], self.store, self.settings)
            except Exception:  # pragma: no cover - defensive
                log.exception("worker crashed on job", extra={"job_id": job_id})
            finally:
                self.queue.task_done()
