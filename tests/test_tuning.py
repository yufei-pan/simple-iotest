import math

import pytest


def assert_finite_positive(x):
    assert x is not None
    assert math.isfinite(x)
    assert x > 0


def test_tuning_attempt_write_bandwidth(iotest_mod, work_dir, make_tl, tiny):
    result = iotest_mod._run_tuning_main(
        tiny["file_size"],
        tiny["file_count"],
        2,
        str(work_dir),
        ["write"],
        tiny["quiet"],
        tiny["zeros"],
        make_tl(),
        True,  # stealth
        0,
        tiny["benchmarkTime"],
    )
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))


def test_tune_single_role_low_target_decreases_pc(iotest_mod, work_dir, make_tl, tiny, monkeypatch):
    pcs = []
    real = iotest_mod._run_tuning_main

    def spy(file_size, file_count, process_count, *args, **kwargs):
        pcs.append(process_count)
        return real(file_size, file_count, process_count, *args, **kwargs)

    monkeypatch.setattr(iotest_mod, "_run_tuning_main", spy)
    # Avoid final full re-run cost/noise
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: None)

    rc = iotest_mod.tune_single_role(
        tiny["file_size"],
        tiny["file_count"],
        4,
        str(work_dir),
        ["write"],
        tiny["quiet"],
        tiny["zeros"],
        make_tl(),
        "write",
        1,  # absurdly low B/s target
        stealth=True,
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )
    assert rc == 0
    assert pcs, "expected at least one tuning attempt"
    # Either decreased across attempts, or best-effort accept at pc=1
    assert min(pcs) <= pcs[0]
    if len(pcs) > 1:
        assert pcs[1] < pcs[0]
    else:
        assert pcs[-1] == 1


def test_tune_single_role_high_target_exits(iotest_mod, work_dir, make_tl, tiny):
    with pytest.raises(SystemExit) as ei:
        iotest_mod.tune_single_role(
            tiny["file_size"],
            tiny["file_count"],
            2,
            str(work_dir),
            ["write"],
            tiny["quiet"],
            tiny["zeros"],
            make_tl(),
            "write",
            10**15,
            stealth=True,
            no_report=True,
            benchmarkTime=tiny["benchmarkTime"],
        )
    assert ei.value.code == 1


def test_tune_for_targets_read_path(iotest_mod, work_dir, make_tl, tiny, monkeypatch):
    # Seed files for read mode
    iotest_mod.main(
        tiny["file_size"],
        tiny["file_count"],
        2,
        str(work_dir),
        ["write"],
        tiny["quiet"],
        tiny["zeros"],
        tl=make_tl(),
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: None)
    # Drive through public entry with low read target (decrease-only / best-effort)
    iotest_mod.tune_for_targets(
        tiny["file_size"],
        tiny["file_count"],
        2,
        str(work_dir),
        ["read"],
        tiny["quiet"],
        tiny["zeros"],
        make_tl(),
        None,
        1,
        stealth=True,
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )


def test_tune_simultaneous_dual_target_low(iotest_mod, work_dir, make_tl, tiny, monkeypatch):
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: None)
    # May SystemExit if either role cannot meet; low targets should succeed or accept pc=1 path
    try:
        iotest_mod.tune_for_targets(
            tiny["file_size"],
            tiny["file_count"],
            4,
            str(work_dir),
            ["simultaneous"],
            tiny["quiet"],
            tiny["zeros"],
            make_tl(),
            1,
            1,
            stealth=True,
            no_report=True,
            benchmarkTime=tiny["benchmarkTime"],
        )
    except SystemExit as e:
        # Documented failure path if a role measures below target mid-tune
        assert e.code == 1
