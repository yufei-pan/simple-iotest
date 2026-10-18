import json
import subprocess
import sys

from test_cli import MODULE

RESULT_KEYS = {
    "ops", "failed", "first_error", "process_count", "total_bytes", "bytes_per_s_call",
    "bytes_per_s_sync", "io_time_pct", "iops_call", "iops_sync", "latency_s", "sync_estimated",
    "late_workers", "max_late_s",
}
LATENCY_KEYS = {"avg", "median", "p1", "p99", "p99_9", "max"}


def test_json_summary_for_write_and_read(run_main, rec_tl, iotest_mod, tmp_path, work_dir):
    path = tmp_path / "out.json"
    result = run_main(["write", "read", "index"], tl=rec_tl, json_path=str(path))
    assert result["report_errors"] == []
    data = json.loads(path.read_text())
    assert set(data) == {"tool", "version", "host", "started", "options", "results", "failed_ops", "report_errors"}
    assert data["tool"] == "iotest"
    assert data["version"] == iotest_mod.version
    assert data["failed_ops"] == 0 and data["report_errors"] == []
    assert len(data["started"]) == 25 and data["started"][-6] in "+-"
    assert data["options"] == {
        "file_size": 1024, "file_count": 1, "process_count": 1, "directories": [str(work_dir)],
        "modes": ["write", "read", "index"], "zeros": True, "unique": False, "start_at": None,
        "block_size": None,
    }
    assert list(data["results"]) == ["write", "read", "index"]
    for key in ("write", "read"):
        entry = data["results"][key]
        assert set(entry) == RESULT_KEYS
        assert set(entry["latency_s"]) == LATENCY_KEYS
        assert entry["ops"] == 1 and entry["failed"] == 0 and entry["first_error"] is None
        assert entry["total_bytes"] == 1024
        assert isinstance(entry["bytes_per_s_call"], float)
        assert isinstance(entry["bytes_per_s_sync"], float)
        assert isinstance(entry["io_time_pct"], float)
        assert entry["sync_estimated"] is False and entry["late_workers"] == 0
    index = data["results"]["index"]
    assert index["total_bytes"] is None and index["bytes_per_s_call"] is None
    assert index["io_time_pct"] is None and isinstance(index["iops_call"], float)


def test_json_is_written_with_no_report_and_names_failures(run_main, rec_tl, tmp_path, work_dir):
    (work_dir / "0" / "temp_1.bin").mkdir(parents=True)
    path = tmp_path / "out.json"
    run_main(["write"], tl=rec_tl, no_report=True, json_path=str(path))
    data = json.loads(path.read_text())
    assert data["failed_ops"] == 1
    entry = data["results"]["write"]
    assert entry["ops"] == 0 and entry["failed"] == 1 and entry["first_error"]
    assert entry["latency_s"] is None and entry["total_bytes"] == 0


def test_unwritable_json_path_is_a_report_error(run_main, rec_tl, tmp_path):
    result = run_main(["write"], tl=rec_tl, json_path=str(tmp_path / "missing" / "out.json"))
    assert len(result["report_errors"]) == 1
    assert result["report_errors"][0].startswith("Could not write ")
    assert any("Report for write mode" in line for line in rec_tl.printed)


def cli(tmp_path, *extra):
    work = tmp_path / "w"
    work.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, str(MODULE), "-fs", "1k", "-fc", "10", "-pc", "1", "-d", str(work),
         "-nl", "-z", "-bt", "1", *extra, "write"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False,
    )


def test_cli_json_to_stdout_with_stealth_is_only_json(tmp_path):
    proc = cli(tmp_path, "-S", "-j", "-")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["results"]["write"]["ops"] == 10


def test_cli_unwritable_json_path_exits_1(tmp_path):
    proc = cli(tmp_path, "-nr", "--json", str(tmp_path / "missing" / "out.json"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Report for write mode" in proc.stdout


def test_cli_json_file_with_no_report(tmp_path):
    path = tmp_path / "out.json"
    proc = cli(tmp_path, "-nr", "-j", str(path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(path.read_text())["results"]["write"]["ops"] == 10


def test_format_key_report_on_fixed_statistics(iotest_mod):
    stats = {
        "ops": 4, "failed": 1, "first_error": "boom", "process_count": 2, "total_bytes": 4096,
        "bytes_per_s_call": 2048.0, "bytes_per_s_sync": 1024.0, "io_time_pct": 50.0,
        "iops_call": 2.0, "iops_sync": 1.0,
        "latency_s": {"avg": 1.0, "median": 2.0, "p1": 0.5, "p99": 3.0, "p99_9": 3.5, "max": 4.0},
    }
    call = iotest_mod.format_bytes(2048.0)
    sync = iotest_mod.format_bytes(1024.0)
    call_bits = iotest_mod.format_bytes(2048.0 * 8, use_1024_bytes=False)
    sync_bits = iotest_mod.format_bytes(1024.0 * 8, use_1024_bytes=False)
    assert iotest_mod.format_key_report("write", stats, (1, 0.25), True, False, []) == [
        "*" * 80,
        "Report for write mode:",
        "Failed write ops:      \t1 of 5, first error: boom",
        "Late starts:           \t1 of 2 workers, up to 0.2500 s late (sync figures include the delay)",
        "Total write size:      \t" + iotest_mod.format_bytes(4096) + "B",
        "write bandwidth (call):\t" + call + "B/s",
        "                       \t" + call_bits + "bps",
        "write bandwidth (sync):\t" + sync + "B/s",
        "                       \t" + sync_bits + "bps",
        "IO time % (IO wait):   \t50.00%",
        "Total write ops:       \t4",
        "write IOPS (call):     \t2.00",
        "write IOPS (sync):     \t1.00",
        "Average write time:    \t1.0000 s",
        "Median write time:     \t2.0000 s",
        "1 % low write time:    \t0.5000 s",
        "1 % high write time:   \t3.0000 s",
        "0.1 % high write time: \t3.5000 s",
        "Highest write time:    \t4.0000 s",
        "Note: sync figures are estimates: comprehensive mode splits the pass time across write/move/stat/read by their share of op time.",
        "-" * 80,
    ]


def test_summarize_key_does_not_reorder_its_input(iotest_mod):
    times = [0.3, 0.1, 0.2, 0.5, 0.4]
    stats = iotest_mod.summarize_key(times, [], 1, 1.0, 1024, True)
    assert times == [0.3, 0.1, 0.2, 0.5, 0.4]
    assert stats["latency_s"]["median"] == 0.3 and stats["latency_s"]["max"] == 0.5


def test_csv_column_matches_arrival_order_of_returned_results(run_main, rec_tl, work_dir):
    result = run_main(["write"], tl=rec_tl, no_report=False, file_count=20)
    times = result["outResults"]["write"]
    csv_files = [p for p in work_dir.glob("iotest_*.csv") if not p.name.endswith("_ops.csv")]
    assert len(csv_files) == 1
    lines = csv_files[0].read_text().splitlines()
    assert [float(v) for v in lines[1:]] == times


def test_cli_tuning_with_json_stdout_and_stealth_is_only_json(tmp_path):
    proc = cli(tmp_path, "-S", "-j", "-", "-wtpt", "1k")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["results"]["write"]["ops"] == 10
