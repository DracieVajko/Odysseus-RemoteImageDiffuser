"""Concurrency + regression tests for unified.model_manager."""
import threading
import time

from unified.model_manager import ModelManager


def test_basic_switch_and_unload():
    mm = ModelManager()
    calls = []
    mm.on_unload = lambda reason: calls.append(reason)

    assert mm.ensure("image", "A") == "A"
    assert mm.status()["loaded_id"] == "A"
    assert mm.ensure("image", "A") == "A"          # same model: no switch
    assert calls == []
    assert mm.ensure("audio", "B") == "B"          # switch: one unload
    assert calls == ["switch"]
    assert mm.status()["loaded_id"] == "B"
    assert mm.unload("idle") is True
    assert calls == ["switch", "idle"]
    assert mm.status()["loaded_id"] is None


def test_switch_releases_before_new_state_visible():
    """A concurrent ensure must block until on_unload finishes."""
    mm = ModelManager()
    callback_started = threading.Event()
    callback_ended = threading.Event()

    def slow_unload(reason):
        callback_started.set()
        time.sleep(0.25)
        callback_ended.set()

    mm.on_unload = slow_unload
    mm.ensure("image", "A")

    t_holder = {}

    def switch_to_b():
        t0 = time.time()
        mm.ensure("audio", "B")
        t_holder["b_done"] = time.time()

    th = threading.Thread(target=switch_to_b)
    th.start()
    assert callback_started.wait(1.0), "on_unload was not called"
    # A second ensure while the callback runs must block until it ends.
    t1 = time.time()
    mm.ensure("vlm", "C")
    t2 = time.time()
    th.join(2.0)

    assert callback_ended.is_set()
    # The second ensure waited for the callback (>= ~0.2 s remaining).
    assert (t2 - t1) >= 0.15, f"ensure did not block on the release ({(t2 - t1):.3f}s)"
    assert mm.status()["loaded_id"] == "C"


def test_begin_end_updates_busy_and_queue():
    """busy/queue must follow the real in-flight count, not stay frozen."""
    mm = ModelManager()
    assert mm.status()["busy"] is False
    assert mm.status()["queue"] == 0

    mm.begin()
    assert mm.status()["busy"] is True
    assert mm.status()["queue"] == 0
    assert mm.in_flight() == 1

    mm.begin()
    assert mm.status()["busy"] is True
    assert mm.status()["queue"] == 1
    assert mm.in_flight() == 2

    mm.end()
    assert mm.status()["busy"] is True
    assert mm.status()["queue"] == 0
    assert mm.in_flight() == 1

    mm.end()
    assert mm.status()["busy"] is False
    assert mm.status()["queue"] == 0
    assert mm.in_flight() == 0

    # Unbalanced end (error path) must not go negative.
    mm.end()
    assert mm.status()["busy"] is False
    assert mm.status()["queue"] == 0
    assert mm.in_flight() == 0


def test_concurrent_ensures_no_deadlock_and_consistent_state():
    mm = ModelManager()
    calls = []
    lock = threading.Lock()

    def on_unload(reason):
        with lock:
            calls.append(reason)

    mm.on_unload = on_unload
    kinds = ["image", "audio", "vlm", "upscale"]
    errors = []

    def worker(i):
        try:
            for j in range(10):
                kind = kinds[(i + j) % len(kinds)]
                mm.ensure(kind, f"{kind}-{i}-{j}")
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)
    elapsed = time.time() - t0

    assert not errors, errors
    assert all(not t.is_alive() for t in threads), "deadlock: worker still alive"
    st = mm.status()
    assert st["loaded_kind"] in kinds
    assert st["loaded_id"]
    # Every switch releases exactly once; load_count counts every switch.
    assert len(calls) == mm.load_count - 1, (len(calls), mm.load_count)
    assert elapsed < 5.0, f"too slow: {elapsed:.1f}s"


if __name__ == "__main__":
    test_basic_switch_and_unload()
    test_switch_releases_before_new_state_visible()
    test_begin_end_updates_busy_and_queue()
    test_concurrent_ensures_no_deadlock_and_consistent_state()
    print("model_manager concurrency tests: PASS")
