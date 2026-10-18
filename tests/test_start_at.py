import json
import subprocess
import sys
import time
from pathlib import Path

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
        before = (time.perf_counter(), time.time())
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
    barrier, (perf_before, wall_before) = seen[0]
    assert barrier - perf_before >= start_at - wall_before - 0.01
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
