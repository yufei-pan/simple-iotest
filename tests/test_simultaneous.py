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


def test_tuning_attempt_readers_find_the_padded_layout(iotest_mod, work_dir, rec_tl):
    # the tuner lays out read files for its largest process count (12 -> worker_00..worker_10)
    # and then runs attempts at smaller counts; readers at pc=9 must still use worker_00..,
    # not worker_0.., or they recreate every read file inside the timed read pass
    iotest_mod.run_simultaneous_layout(1, 1024, [str(work_dir)], "r", 11, 12, True, True, rec_tl)
    iotest_mod.main(1024, 1, 9, str(work_dir), ["simultaneous"], True, True, tl=rec_tl, no_report=True,
        benchmarkTime=1, simultaneous_run_id="r", simultaneous_skip_layout=True,
        simultaneous_skip_cleanup=True, simultaneous_layout_process_count=12, simultaneous_write_count=4)
    simul = Path(work_dir) / ".iotest_simul_r"
    # writers use the layout's padding too, so every attempt overwrites the same files
    # instead of leaving a second set in worker_0.. when the count drops below 10
    assert list(simul.glob("worker_?")) == []
