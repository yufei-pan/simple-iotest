import json
import math
import sys
from pathlib import Path

import pytest


def assert_finite_positive(x):
    assert x is not None
    assert math.isfinite(x)
    assert x > 0


def test_simultaneous_cleans_up_and_reports_bandwidth(run_main, work_dir, iotest_mod, make_tl):
    result = run_main(
        ["simultaneous"],
        process_count=2,
        tl=make_tl(),
    )
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))
    leftovers = list(Path(work_dir).glob(".iotest_simul_*"))
    assert leftovers == []


def test_simultaneous_multi_directory(iotest_mod, tmp_path, make_tl, tiny):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    result = iotest_mod.main(
        tiny["file_size"],
        tiny["file_count"],
        2,
        [str(d1), str(d2)],
        ["simultaneous"],
        tiny["quiet"],
        tiny["zeros"],
        tl=make_tl(),
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))
    assert list(d1.glob(".iotest_simul_*")) == []
    assert list(d2.glob(".iotest_simul_*")) == []


def test_simultaneous_after_batch_write_uses_role_worker_counts(run_main, iotest_mod):
    # batch write stores "write", then simultaneous collides and both become write_1/write_2;
    # write_2 must keep the simultaneous writer count, not fall back to process_count
    result = run_main(["write", "simultaneous"], process_count=3, file_count=2)
    assert set(result["outResults"]) == {"write_1", "write_2", "read"}
    assert result["mode_process_counts"] == {"write_1": 3, "write_2": 1, "read": 2}
    for mode in result["outResults"]:
        assert len(result["outResults"][mode]) == 2 * result["mode_process_counts"][mode]
    expected = (1024 * 2 * 1) / result["totalTime"]["write_2"]
    assert iotest_mod.measure_role_bandwidth(result, "write_2") == pytest.approx(expected)


def test_simultaneous_requires_pc_ge_2(run_main):
    with pytest.raises(ValueError, match="process_count >= 2"):
        run_main(["simultaneous"], process_count=1)


def test_tuning_attempt_readers_find_the_layout_of_a_larger_count(iotest_mod, work_dir, rec_tl):
    # the tuner lays out read files for its largest process count (12 -> 11 layout workers) and then
    # runs attempts at smaller counts; readers at pc=9 must find those files, or they recreate every
    # read file inside the timed read pass
    iotest_mod.run_simultaneous_layout(1, 1024, [str(work_dir)], "r", 11, True, True, rec_tl)
    simul = Path(work_dir) / ".iotest_simul_r"
    laid_out = {p: p.stat().st_mtime_ns for p in simul.rglob("read_*.bin")}
    iotest_mod.main(1024, 1, 9, str(work_dir), ["simultaneous"], True, True, tl=rec_tl, no_report=True,
        benchmarkTime=1, simultaneous_run_id="r", simultaneous_skip_layout=True,
        simultaneous_skip_cleanup=True, simultaneous_write_count=4)
    assert {p: p.stat().st_mtime_ns for p in simul.rglob("read_*.bin")} == laid_out
    # writers are the first workers too, so every attempt overwrites the same files
    assert sorted(p.name for p in simul.iterdir()) == [f"worker_{c:04d}" for c in range(11)]


# what writing to a directory raises: Windows refuses it as a permission error
BLOCKED_ERROR = "PermissionError" if sys.platform == "win32" else "IsADirectoryError"


def block_layout(iotest_mod, work_dir, run_id, counters, name="read_1.bin"):
    # a directory where a read file belongs makes that layout write fail
    for counter in counters:
        worker_dir = Path(iotest_mod.simultaneous_worker_dir(str(work_dir), run_id, iotest_mod.worker_counter_name(counter)))
        (worker_dir / name).mkdir(parents=True)


def test_layout_write_failure_skips_the_timed_pass_and_is_reported(iotest_mod, run_main, work_dir, rec_tl):
    block_layout(iotest_mod, work_dir, "lf", range(2))
    result = run_main(["simultaneous"], process_count=2, tl=rec_tl, simultaneous_run_id="lf")
    assert "layout_errors" not in result
    assert any("Layout pass could not write 2 read files" in line for line in rec_tl.printed)
    assert not any("writers +" in line for line in rec_tl.printed)
    # under a key of their own, with no times: layout writes are not timed
    assert result["outResults"] == {"layout": []}
    assert len(result["failedOps"]["layout"]) == 2
    assert all(BLOCKED_ERROR in error for error in result["failedOps"]["layout"])
    assert result["mode_process_counts"] == {"layout": 2}
    assert iotest_mod.total_failed_ops(result) == 2
    assert not (Path(work_dir) / ".iotest_simul_lf").exists()


def test_a_failed_layout_counts_even_when_a_later_one_succeeds(iotest_mod, run_main, work_dir, rec_tl):
    # the blockers go with the first pass's tree, so only its layout fails; the run must still exit 1
    block_layout(iotest_mod, work_dir, "lf", range(2))
    result = run_main(["simultaneous", "simultaneous"], process_count=2, tl=rec_tl, simultaneous_run_id="lf")
    assert set(result["outResults"]) == {"layout", "write", "read"}
    assert len(result["failedOps"]["layout"]) == 2
    assert iotest_mod.total_failed_ops(result) == 2


def test_failed_layouts_of_several_passes_accumulate(iotest_mod, run_main, work_dir, rec_tl):
    block_layout(iotest_mod, work_dir, "lf", range(2))
    result = run_main(["simultaneous", "simultaneous"], process_count=2, tl=rec_tl, simultaneous_run_id="lf",
        simultaneous_skip_cleanup=True)
    assert result["outResults"] == {"layout_1": [], "layout_2": []}
    assert {key: len(errors) for key, errors in result["failedOps"].items()} == {"layout_1": 2, "layout_2": 2}
    assert iotest_mod.total_failed_ops(result) == 4


def test_a_partly_failed_layout_reports_its_failures_everywhere(iotest_mod, run_main, work_dir, rec_tl, tmp_path, webhook):
    # one of the 4 read files cannot be written
    block_layout(iotest_mod, work_dir, "lf", [0])
    url, received = webhook()
    json_path = tmp_path / "out.json"
    result = run_main(["simultaneous"], process_count=2, file_count=2, tl=rec_tl, simultaneous_run_id="lf",
        no_report=False, json_path=str(json_path), message_end_point_address=url)
    assert iotest_mod.total_failed_ops(result) == 1
    report = "\n".join(rec_tl.printed)
    assert "Report for layout mode:" in report
    assert "Failed layout ops:      \t1 of 4, first error: " + BLOCKED_ERROR in report
    assert "Layout writes are not timed, and the timed simultaneous pass did not run." in report
    data = json.loads(json_path.read_text())
    assert data["failed_ops"] == 1
    entry = data["results"]["layout"]
    assert entry["ops"] == 0 and entry["failed"] == 1 and entry["process_count"] == 2
    assert entry["first_error"].startswith(BLOCKED_ERROR)
    assert entry["latency_s"] is None and entry["total_bytes"] is None
    (report_file,) = work_dir.glob("*_report.txt")
    assert "Failed layout ops:      \t1 of 4" in report_file.read_text()
    (csv_file,) = work_dir.glob("iotest_*.csv")
    assert csv_file.read_text().splitlines() == ["layout_time"]
    (request,) = received()
    assert request["json"]["content"].startswith("Short Report for layout mode: 1 of 4 ops failed, first error: " + BLOCKED_ERROR)


@pytest.mark.patches_workers
def test_a_layout_worker_that_dies_fails_all_its_files_once(iotest_mod, run_main, rec_tl, monkeypatch):
    # worker 0001 fails its first file, then dies before handing its failures over (a MemoryError making its
    # content, say): all 3 of its files count as failed, the first not twice
    real_write = iotest_mod.FileContent.write

    def write(self, file_name, index, quiet=False):
        if "worker_0001" in file_name:
            if index == 0:
                return None, "OSError: first file"
            raise MemoryError
        return real_write(self, file_name, index, quiet)

    monkeypatch.setattr(iotest_mod.FileContent, "write", write)
    result = run_main(["simultaneous"], process_count=2, file_count=3, tl=rec_tl)
    assert result["outResults"] == {"layout": []}
    assert result["failedOps"]["layout"] == ["worker process exited with code 1 before reporting results"] * 3
    assert iotest_mod.total_failed_ops(result) == 3


def test_the_layout_uses_the_runs_manager(iotest_mod, run_main, monkeypatch):
    real = iotest_mod.Manager
    made = []
    monkeypatch.setattr(iotest_mod, "Manager", lambda: made.append(1) or real())
    result = run_main(["simultaneous"], process_count=2)
    assert iotest_mod.total_failed_ops(result) == 0
    assert len(made) == 1


def test_simultaneous_tuning_exits_1_when_its_layout_fails(iotest_mod, work_dir, rec_tl):
    # the tuner lays out under the logger's datetime, with max(pc - 1, 1) workers
    block_layout(iotest_mod, work_dir, rec_tl.currentDateTime, [0])
    options = iotest_mod.RunOptions(file_size=1024, file_count=1, process_count=2, directories=[str(work_dir)],
        modes=["simultaneous"], zeros=True, no_report=True, benchmark_time=1)
    with pytest.raises(SystemExit) as exit_info:
        iotest_mod.tune_for_targets(options, rec_tl, 1, None)
    assert exit_info.value.code == 1
    assert any("Layout pass could not write 1 read files" in line for line in rec_tl.printed)
    assert not (Path(work_dir) / f".iotest_simul_{rec_tl.currentDateTime}").exists()
