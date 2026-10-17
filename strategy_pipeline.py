"""Overlap remote planning with main-thread coder startup and generation."""

from queue import Empty, Queue
import threading


class StrategyArrivalStream:
    """Run API rounds in a producer thread; publish only accepted slots.

    GPU lifecycle, tokenizer rendering, rollout bookkeeping, and evaluation
    submission remain on the consuming thread. The producer exclusively owns
    strategy history and its existing retry/round-barrier policy.
    """

    def __init__(self, run):
        self._events = Queue()
        self._stop = threading.Event()
        self._done = False

        def produce():
            try:
                run(self._publish, self._stop)
            except BaseException as exc:
                self._events.put(("error", exc))
            finally:
                self._events.put(("done", None))

        self._thread = threading.Thread(
            target=produce, name="api-strategy-producer", daemon=True)

    def _publish(self, parent, strategy):
        if not self._stop.is_set():
            self._events.put(("strategy", (int(parent), int(strategy))))

    def __enter__(self):
        self._thread.start()
        return self

    def poll(self):
        ready = []
        while True:
            try:
                kind, value = self._events.get_nowait()
            except Empty:
                break
            if kind == "error":
                raise value
            if kind == "done":
                self._done = True
            else:
                ready.append(value)
        return ready, self._done

    def __exit__(self, *_exc):
        self._stop.set()
        # A synchronous HTTP request already in flight may finish later. Its
        # cancellation gate prevents further history writes or new rounds.
        self._thread.join(timeout=1.0)


def iter_ready_batches(poll_jobs, run_batch):
    """Compatibility path for local/HF coders without a live worker scheduler.

    Drain all currently available jobs into ordinary cross-prompt batches.
    Remote planning continues while each batch generates and evaluates.
    """
    wake = threading.Event()
    done = False
    while not done:
        jobs, done = poll_jobs()
        if not jobs:
            if not done:
                wake.wait(0.05)
            continue
        # A task has one generation seed step. Keep pilot/adaptive seeds apart.
        by_step = {}
        for job in jobs:
            by_step.setdefault(job["step_idx"], []).append(job)
        for batch in by_step.values():
            yield from run_batch(batch)
