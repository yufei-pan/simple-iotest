import math
import multiprocessing.managers
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


# Anomaly gate: with a single file, outResults has one entry so the 1% high and
# 1% low samples are the same value x, and the condition is `x > x * threshold`.
# That makes the branch deterministic: threshold < 1 always fires, >= 1 never does.


def _run_with_anomaly_threshold(iotest_mod, work_dir, rec_tl, threshold):
    return iotest_mod.main(
        1024,  # file_size
        1,     # file_count
        1,     # process_count
        str(work_dir),
        ["write"],
        True,  # quiet
        True,  # zeros
        tl=rec_tl,
        no_report=True,
        threshold_to_report_anomaly=threshold,
        benchmarkTime=1,
    )


def test_anomaly_warning_reported_for_sub_unit_threshold(iotest_mod, work_dir, rec_tl):
    _run_with_anomaly_threshold(iotest_mod, work_dir, rec_tl, 0.5)
    report = "\n".join(rec_tl.printed)
    assert "1% high is too high compared to 1% low" in report
    assert "threshold is 0.5" in report


def test_anomaly_warning_not_reported_at_threshold_one(iotest_mod, work_dir, rec_tl):
    # threshold=1 with a single sample: high == low so x > x*1 is false
    _run_with_anomaly_threshold(iotest_mod, work_dir, rec_tl, 1)
    report = "\n".join(rec_tl.printed)
    assert "1% high is too high compared to 1% low" not in report


def test_anomaly_disabled_at_zero_threshold(iotest_mod, work_dir, rec_tl):
    _run_with_anomaly_threshold(iotest_mod, work_dir, rec_tl, 0)
    report = "\n".join(rec_tl.printed)
    assert "1% high is too high compared to 1% low" not in report


def _report_value(report, label):
    for line in report.splitlines():
        if line.startswith(label):
            return line.split("\t", 1)[1].strip()
    raise AssertionError(f"{label!r} not found in report")


def _run_report(iotest_mod, work_dir, rec_tl, modes, file_count=3, process_count=2):
    iotest_mod.main(1024, file_count, process_count, str(work_dir), modes, True, True,
                    tl=rec_tl, no_report=True, benchmarkTime=1)
    return "\n".join(rec_tl.printed)


@pytest.mark.parametrize(
    "modes, roles",
    [
        (["index"], ["index"]),
        (["write"], ["write"]),
        (["comprehensive"], ["write", "move", "stat", "read"]),
    ],
)
def test_report_includes_iops(iotest_mod, work_dir, rec_tl, modes, roles):
    report = _run_report(iotest_mod, work_dir, rec_tl, modes)
    for role in roles:
        # 3 files per process * 2 processes
        assert _report_value(report, f"Total {role} ops:") == "6"
        for kind in ("call", "sync"):
            iops = float(_report_value(report, f"{role} IOPS ({kind}):").replace(",", ""))
            assert_finite_positive(iops)


def test_parent_reads_each_manager_list_in_one_call(run_main, rec_tl, monkeypatch):
    # a proxy read item by item costs one round trip to the manager per op
    calls = []
    real = multiprocessing.managers.BaseProxy._callmethod

    def spy(self, methodname, args=(), kwds={}):
        calls.append((methodname, args))
        return real(self, methodname, args, kwds)

    monkeypatch.setattr(multiprocessing.managers.BaseProxy, "_callmethod", spy)
    run_main(["write", "random", "comprehensive", "simultaneous", "meta"], process_count=2, file_count=3, tl=rec_tl,
             timeline=True)
    reads = [args for name, args in calls if name == "__getitem__"]
    # generation benchmark; write, random and comprehensive: results and late starts, and the timeline;
    # simultaneous: the layout's errors, both roles' results and late starts, and the timeline; meta: 3 passes of 3
    assert len(reads) == 1 + 3 * 3 + 6 + 3 * 3
    assert all(args == (slice(None, None, None),) for args in reads)
    assert not [name for name, _ in calls if name in ("__len__", "__iter__", "__contains__")]
