import math
from pathlib import Path

import pytest


def assert_finite_positive(x):
    assert x is not None
    assert math.isfinite(x)
    assert x > 0


def test_write_mode_leaves_files_and_positive_bandwidth(run_main, work_dir, iotest_mod):
    result = run_main(["write"])
    assert "write" in result["outResults"]
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    # workers write under <work_dir>/<counter>/temp_*.bin, not the top level
    files = list(Path(work_dir).rglob("*"))
    assert any(p.is_file() for p in files)


def test_read_mode_after_write(run_main, work_dir, iotest_mod):
    run_main(["write"])
    result = run_main(["read"])
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))


def test_index_mode(run_main):
    result = run_main(["index"])
    assert "index" in result["outResults"]


def test_comprehensive_mode(run_main, iotest_mod):
    result = run_main(["comprehensive"])
    # comprehensive stores write/move/stat/read timing keys — assert no crash and write/read if present
    assert result["outResults"]
    for role in ("write", "read"):
        if role in result["outResults"] and role in result["totalTime"]:
            assert_finite_positive(iotest_mod.measure_role_bandwidth(result, role))


def test_random_mode(run_main):
    result = run_main(["random"])
    assert result["outResults"]


def test_benchmark_mode(run_main):
    result = run_main(["benchmark"], benchmarkTime=1)
    # benchmark skips worker modes; still returns dict
    assert isinstance(result, dict)
    assert result["process_count"] >= 1


def test_rw_batch_modes(run_main, iotest_mod):
    modes = iotest_mod.parse_modes(["rw"])
    result = run_main(modes)
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))


def test_rwi_batch_modes(run_main, iotest_mod):
    modes = iotest_mod.parse_modes(["rwi"])
    result = run_main(modes)
    assert "index" in result["outResults"]
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))


def test_threshold_to_report_anomaly_smoke(run_main):
    result = run_main(["write"], threshold_to_report_anomaly=1)
    assert "write" in result["outResults"]
