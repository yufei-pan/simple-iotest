import math
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


def test_layout_write_failure_skips_the_timed_pass_and_is_reported(iotest_mod, run_main, work_dir, rec_tl):
    # a directory where a read file belongs makes that layout write fail
    for counter in range(2):
        blocker = Path(iotest_mod.simultaneous_worker_dir(str(work_dir), "lf", iotest_mod.worker_counter_name(counter))) / "read_1.bin"
        blocker.mkdir(parents=True)
    result = run_main(["simultaneous"], process_count=2, tl=rec_tl, simultaneous_run_id="lf")
    assert len(result["layout_errors"]) == 2
    assert any("Layout pass could not write 2 read files" in line for line in rec_tl.printed)
    assert not any("writers +" in line for line in rec_tl.printed)
    assert result["outResults"] == {}
    assert iotest_mod.total_failed_ops(result) == 2
    assert not (Path(work_dir) / ".iotest_simul_lf").exists()
