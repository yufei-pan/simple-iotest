import array
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import pytest

import iotest

MODULE = Path(iotest.__file__)


def run_cli(args):
    return subprocess.run(
        [sys.executable, str(MODULE), "-nl", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, check=False,
    )


def spy_barriers(monkeypatch):
    seen = []
    real = iotest.first_pass_start

    def spy(*args):
        before = (iotest.shared_clock(), time.time())
        barrier = real(*args)
        seen.append((barrier, before))
        return barrier

    monkeypatch.setattr(iotest, "first_pass_start", spy)
    return seen


def test_start_at_holds_first_pass_until_the_time(run_main, rec_tl, monkeypatch):
    seen = spy_barriers(monkeypatch)
    start_at = time.time() + 4
    run_main(["write", "read"], tl=rec_tl, start_at=start_at)
    # one synchronized pass; the second pass schedules itself as usual
    assert len(seen) == 1
    barrier, (clock_before, wall_before) = seen[0]
    assert barrier - clock_before >= start_at - wall_before - 0.01
    assert time.time() >= start_at
    assert any(p.startswith("Synchronized start at ") and p.endswith("(--start_at)") for p in rec_tl.printed[-1].splitlines())
    assert not any("too soon" in m for m in rec_tl.logged)


def test_start_at_covers_simultaneous_after_layout(run_main, rec_tl, monkeypatch):
    seen = spy_barriers(monkeypatch)
    start_at = time.time() + 4
    run_main(["simultaneous"], tl=rec_tl, start_at=start_at, process_count=2)
    assert len(seen) == 1
    assert time.time() >= start_at


def test_start_at_too_soon_warns_and_reports_lateness(run_main, rec_tl):
    result = run_main(["write"], tl=rec_tl, start_at=time.time() + 0.05)
    assert result["failedOps"] == {"write": []}
    assert any("--start_at is too soon" in m and "s late" in m for m in rec_tl.logged)
    report = "\n".join(rec_tl.printed)
    assert "Synchronized start at " in report and "(--start_at)" in report
    assert "s late: less than the" in report


def test_start_at_in_json_options(run_main, rec_tl, work_dir):
    path = work_dir / "out.json"
    start_at = time.time() + 3
    run_main(["write"], tl=rec_tl, start_at=start_at, json_path=str(path))
    assert json.loads(path.read_text())["options"]["start_at"] == start_at


def test_report_has_no_start_line_without_start_at(run_main, rec_tl):
    run_main(["write"], tl=rec_tl)
    assert "Synchronized start" not in "\n".join(rec_tl.printed)


def test_cli_rejects_past_and_non_numeric_start_at(work_dir):
    for value in (str(time.time() - 10), "soon", "nan", "inf"):
        proc = run_cli(["-d", str(work_dir), "--start_at", value])
        assert proc.returncode == 2, value
        assert "--start_at" in proc.stderr
        assert "Traceback" not in proc.stderr


def test_cli_rejects_start_at_with_throughput_tuning(work_dir):
    proc = run_cli(["-d", str(work_dir), "--start_at", str(time.time() + 60), "-wtpt", "100m"])
    assert proc.returncode == 2
    assert "--start_at cannot be combined with throughput tuning" in proc.stderr


def test_start_at_prints_the_wait_and_json_reports_zero_lateness(run_main, rec_tl, work_dir):
    path = work_dir / "out.json"
    run_main(["write"], tl=rec_tl, start_at=time.time() + 4, json_path=str(path))
    assert any(m.startswith("Waiting until ") and "for --start_at (" in m and m.endswith(" s)") for m in rec_tl.printed)
    assert json.loads(path.read_text())["start_late_s"] == 0.0


def test_start_at_too_soon_json_reports_lateness(run_main, rec_tl, work_dir):
    path = work_dir / "out.json"
    run_main(["write"], tl=rec_tl, start_at=time.time() + 0.05, json_path=str(path))
    assert json.loads(path.read_text())["start_late_s"] > 0
    assert not any(m.startswith("Waiting until ") for m in rec_tl.printed)


def test_barrier_wait_sleeps_once_until_a_second_before(monkeypatch):
    sleeps = []
    real_sleep = time.sleep

    def record(seconds):
        sleeps.append(seconds)
        real_sleep(min(seconds, 0.001))

    clock = {"offset": 0.0}
    real_perf = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real_perf() + clock["offset"])
    monkeypatch.setattr(time, "sleep", lambda s: (record(s), clock.__setitem__("offset", clock["offset"] + s)))
    iotest.wait_for_barrier(time.perf_counter() + 3)
    assert 1.9 < sleeps[0] <= 2.0
    assert len(sleeps) > 1 and all(s == 0.001 for s in sleeps[1:])


def test_worker_turns_the_shared_start_into_its_own_clock(iotest_mod, work_dir, monkeypatch):
    # a spawned worker's perf_counter() can count from its own start (macOS and Windows before Python 3.10), so it
    # gets the start on shared_clock() and moves it onto perf_counter(): here perf_counter() is an hour ahead
    real_perf = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real_perf() + 3600)
    late, sink = [], []
    iotest_mod.worker(1, 1024, str(work_dir), [], "write", "0", True, True, iotest_mod.shared_clock() + 0.3, False,
        False, late, timeline=sink)
    assert late == []
    (record,) = sink
    # the op's start, in seconds from the barrier
    (op_start,) = array.array("d", record[3])
    assert 0 <= op_start < 60


@pytest.mark.parametrize("modes,start_in", [(["write"], None), (["simultaneous"], None), (["write"], 3)])
@pytest.mark.parametrize("parent_offset", [-3600, 120])
def test_spawned_workers_start_on_time_when_the_parents_clock_differs(run_main, rec_tl, monkeypatch, modes, start_in, parent_offset):
    # spawned workers do not see this patch: the parent's perf_counter() runs ahead of theirs (120 s: a failure then
    # takes two minutes, not an hour), as a parent's did on macOS with Python 3.9 (a spawned worker is younger), or
    # behind. 3.71.0's workers waited as long as it was ahead, or started that late. A wait the parent compensates
    # for in its own wall time (it started the pass 120 s late, by its own clock) shows only in how long the run took
    previous = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    real_perf = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real_perf() + parent_offset)
    start_at = None if start_in is None else time.time() + start_in
    began = real_perf()
    try:
        result = run_main(modes, process_count=2, tl=rec_tl, start_at=start_at)
    finally:
        multiprocessing.set_start_method(previous, force=True)
    # a run that starts on time takes seconds (the --start_at case waits 3 of them)
    assert real_perf() - began < 60
    assert result["failedOps"] == {key: [] for key in result["outResults"]}
    for count, longest in result["late_starts"].values():
        assert longest < 60
    for seconds in result["totalTime"].values():
        assert 0 < seconds < 60


def test_a_pass_is_timed_on_perf_counter_whatever_the_shared_clock_counts(run_main, iotest_mod, monkeypatch):
    # shared_clock() only carries the start to the workers. Where it counts time perf_counter() does not (a Mac
    # asleep), the pass's wall time must not: here it jumps an hour once the pass's workers have been started
    # (a read of files that are there starts no other processes)
    run_main(["write"], process_count=2)
    real_start, real_clock, jump = iotest_mod.start_processes, iotest_mod.shared_clock, [0]
    monkeypatch.setattr(iotest_mod, "shared_clock", lambda: real_clock() + jump[0])

    def start(processes):
        real_start(processes)
        jump[0] += 3600

    monkeypatch.setattr(iotest_mod, "start_processes", start)
    result = run_main(["read"], process_count=2)
    assert jump[0] == 3600
    assert result["failedOps"] == {"read": []}
    assert 0 < result["totalTime"]["read"] < 60


def test_shared_clock_is_perf_counter_from_python_3_10(iotest_mod):
    # from 3.10 perf_counter() is one system-wide clock on every platform, so the start is handed on the clock every
    # measurement uses; the kernel's CLOCK_MONOTONIC would count the time a Mac sleeps, perf_counter() does not
    assert (iotest_mod.shared_clock is time.perf_counter) == (sys.version_info >= (3, 10))


def test_shared_clock_without_clock_gettime_is_one_every_process_reads_alike():
    # python.org's macOS installers of 3.7 and 3.8 have no time.clock_gettime, and there monotonic() counts from the
    # process's start (as perf_counter() does before 3.10): the wall clock is the one every process reads alike
    code = ("import time\n"
            "try:\n    del time.clock_gettime\nexcept AttributeError:\n    pass\n"
            "import iotest\n"
            "print(next(name for name in ('perf_counter', 'monotonic', 'time') if getattr(time, name) is iotest.shared_clock))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(MODULE.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, check=False)
    expected = "perf_counter" if sys.version_info >= (3, 10) else "monotonic" if sys.platform == "win32" else "time"
    assert proc.stdout.strip() == expected, proc.stdout + proc.stderr


@pytest.mark.parametrize("seconds", [5, -5])
def test_own_clock_moves_a_shared_time_onto_perf_counter(iotest_mod, monkeypatch, seconds):
    # a time on shared_clock(), as many seconds ahead of (or behind) now, on a perf_counter() that is an hour ahead;
    # the two reads are microseconds apart, so half a second is room for a slow host
    real_perf = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real_perf() + 3600)
    assert iotest_mod.own_clock(iotest_mod.shared_clock() + seconds) == pytest.approx(time.perf_counter() + seconds, abs=0.5)


def test_start_late_is_null_when_no_pass_ran_at_start_at(iotest_mod, run_main, rec_tl, work_dir, tmp_path):
    # the simultaneous layout fails, so its timed pass, the one --start_at was for, never starts: not "on time"
    blocker = Path(iotest_mod.simultaneous_worker_dir(str(work_dir), "lf", iotest_mod.worker_counter_name(0))) / "read_1.bin"
    blocker.mkdir(parents=True)
    path = tmp_path / "out.json"
    result = run_main(["simultaneous"], process_count=2, tl=rec_tl, start_at=time.time() + 60, json_path=str(path),
        simultaneous_run_id="lf")
    assert iotest_mod.total_failed_ops(result) == 1
    data = json.loads(path.read_text())
    assert data["options"]["start_at"] is not None
    assert data["start_late_s"] is None
    assert "Synchronized start" not in "\n".join(rec_tl.printed)
    assert not any(m.startswith("Waiting until ") for m in rec_tl.printed)


def test_start_late_is_null_for_a_benchmark_only_run(run_main, rec_tl, tmp_path):
    path = tmp_path / "out.json"
    run_main(["benchmark"], tl=rec_tl, start_at=time.time() + 60, json_path=str(path))
    assert json.loads(path.read_text())["start_late_s"] is None
