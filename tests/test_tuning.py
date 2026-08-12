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


# ---------------------------------------------------------------------------
# tune_dual_role_shared_pc (shared-pc dual-target tuning, -wtpt + -rtpt on
# write+read / comprehensive). _run_tuning_main and measure_role_bandwidth are
# stubbed so each branch is driven deterministically instead of depending on
# real disk throughput.
# ---------------------------------------------------------------------------


def _stub_runtime(iotest_mod, monkeypatch, process_count, write_bps, read_bps):
    """Stub tuning I/O: canned result + fixed measured bandwidths."""
    monkeypatch.setattr(
        iotest_mod,
        "_run_tuning_main",
        lambda *a, **k: {"process_count": process_count, "outResults": {}, "totalTime": {}},
    )
    monkeypatch.setattr(
        iotest_mod,
        "measure_role_bandwidth",
        lambda result, role: {"write": write_bps, "read": read_bps}[role],
    )


def _call_dual(iotest_mod, work_dir, rec_tl, write_target, read_target, **kw):
    defaults = dict(
        file_size=1024,
        file_count=1,
        process_count=4,
        directory=str(work_dir),
        modes=["write", "read"],
        quiet=True,
        zeros=True,
        tl=rec_tl,
    )
    defaults.update(kw)
    return iotest_mod.tune_dual_role_shared_pc(
        defaults["file_size"], defaults["file_count"], defaults["process_count"],
        defaults["directory"], defaults["modes"], defaults["quiet"], defaults["zeros"],
        defaults["tl"], write_target, read_target,
        stealth=True, no_report=True, benchmarkTime=1,
    )


def test_dual_role_shared_pc_targets_met_finalizes(iotest_mod, work_dir, rec_tl, monkeypatch):
    # measured == target -> status 'ok' -> finalize and return 0
    _stub_runtime(iotest_mod, monkeypatch, 4, 1000, 2000)
    finalized = []
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: finalized.append(a))
    rc = _call_dual(iotest_mod, work_dir, rec_tl, 1000, 2000)
    assert rc == 0
    assert finalized, "expected tuning run to be finalized"


def test_dual_role_shared_pc_best_effort_at_pc1(iotest_mod, work_dir, rec_tl, monkeypatch):
    # measured above target -> both 'above', effective_pc=1 -> accept best effort
    _stub_runtime(iotest_mod, monkeypatch, 1, 1000, 2000)
    finalized = []
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: finalized.append(a))
    rc = _call_dual(iotest_mod, work_dir, rec_tl, 100, 100)
    assert rc == 0
    assert finalized
    assert any("best effort" in m for m in rec_tl.printed)


def test_dual_role_shared_pc_cannot_meet_exits(iotest_mod, work_dir, rec_tl, monkeypatch):
    # measured below target at attempt 1 -> sys.exit(1)
    _stub_runtime(iotest_mod, monkeypatch, 4, 1, 1)
    with pytest.raises(SystemExit) as ei:
        _call_dual(iotest_mod, work_dir, rec_tl, 10**15, 10**15)
    assert ei.value.code == 1


def test_dual_role_shared_pc_cannot_measure_exits(iotest_mod, work_dir, rec_tl, monkeypatch):
    monkeypatch.setattr(
        iotest_mod,
        "_run_tuning_main",
        lambda *a, **k: {"process_count": 4, "outResults": {}, "totalTime": {}},
    )
    monkeypatch.setattr(iotest_mod, "measure_role_bandwidth", lambda result, role: None)
    with pytest.raises(SystemExit) as ei:
        _call_dual(iotest_mod, work_dir, rec_tl, 1000, 2000)
    assert ei.value.code == 1


# ---------------------------------------------------------------------------
# tune_simultaneous (dedicated simultaneous + target tuning). Layout pass and
# tuning I/O are stubbed; the w/r ratio and process-count search is driven with
# fixed measured bandwidth.
# ---------------------------------------------------------------------------


def _call_simul(iotest_mod, work_dir, rec_tl, write_target, read_target, process_count=2):
    return iotest_mod.tune_simultaneous(
        1024, 1, process_count, [str(work_dir)], ["simultaneous"],
        True, True, rec_tl, write_target, read_target,
        stealth=True, no_report=True, benchmarkTime=1,
    )


def _stub_simul(iotest_mod, monkeypatch, measure, process_count=2):
    """Stub simultaneous tuning I/O; `measure` is the fake bandwidth function."""
    monkeypatch.setattr(iotest_mod, "run_simultaneous_layout", lambda *a, **k: None)
    monkeypatch.setattr(
        iotest_mod,
        "_run_tuning_main",
        lambda *a, **k: {"process_count": process_count, "outResults": {}, "totalTime": {}},
    )
    if callable(measure):
        monkeypatch.setattr(iotest_mod, "measure_role_bandwidth", measure)
    else:
        monkeypatch.setattr(
            iotest_mod, "measure_role_bandwidth",
            lambda result, role: measure[role],
        )
    finalized = []
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: finalized.append(a))
    return finalized


def test_simul_tuning_both_targets_met_finalizes(iotest_mod, work_dir, rec_tl, monkeypatch):
    # measured == target -> status 'ok' for both -> finalize and return 0
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 1000, "read": 2000})
    rc = _call_simul(iotest_mod, work_dir, rec_tl, 1000, 2000)
    assert rc == 0
    assert finalized, "expected simultaneous tuning to finalize"


def test_simul_tuning_both_ok_above_pc2_walks_pc_down(iotest_mod, work_dir, rec_tl, monkeypatch):
    # both 'ok' but pc=4 -> keep reducing pc until pc<=2, then finalize
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 1000, "read": 2000})
    rc = _call_simul(iotest_mod, work_dir, rec_tl, 1000, 2000, process_count=4)
    assert rc == 0
    assert finalized


def test_simul_tuning_cannot_meet_both_exits(iotest_mod, work_dir, rec_tl, monkeypatch):
    _stub_simul(iotest_mod, monkeypatch, {"write": 1, "read": 1})
    with pytest.raises(SystemExit) as ei:
        _call_simul(iotest_mod, work_dir, rec_tl, 10**15, 10**15)
    assert ei.value.code == 1


def test_simul_tuning_write_only_target_reached(iotest_mod, work_dir, rec_tl, monkeypatch):
    # write-only target met -> finalize and return 0
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 1000, "read": 2000})
    rc = _call_simul(iotest_mod, work_dir, rec_tl, 1000, None)
    assert rc == 0
    assert finalized


def test_simul_tuning_read_only_target_reached(iotest_mod, work_dir, rec_tl, monkeypatch):
    # read-only target met -> finalize and return 0 (mirrors write-only branch)
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 1000, "read": 2000})
    rc = _call_simul(iotest_mod, work_dir, rec_tl, None, 2000)
    assert rc == 0
    assert finalized


def test_simul_tuning_write_only_below_accepts_last_valid(iotest_mod, work_dir, rec_tl, monkeypatch, tiny):
    # attempt 1: write above (records last valid config), attempt 2: write below
    # -> cannot reduce further -> accept_last_valid True -> finalize
    values = {"write": [2000, 500], "read": [2000, 2000]}
    counters = {"write": 0, "read": 0}

    def stateful(result, role):
        i = counters[role]
        counters[role] = i + 1
        return values[role][min(i, len(values[role]) - 1)]

    finalized = _stub_simul(iotest_mod, monkeypatch, stateful, process_count=4)
    rc = _call_simul(iotest_mod, work_dir, rec_tl, 1000, None, process_count=4)
    assert rc == 0
    assert finalized


def test_simul_tuning_write_only_above_walks_to_minimum(iotest_mod, work_dir, rec_tl, monkeypatch):
    # write-only, write above target at pc=4 -> reduce write_count then pc via
    # reset_split_for_pc else-branch until minimum -> accept best effort
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 10**9, "read": 2000}, process_count=4)
    rc = _call_simul(iotest_mod, work_dir, rec_tl, 1000, None, process_count=4)
    assert rc == 0
    assert finalized


def test_simul_tuning_read_only_above_walks_to_minimum(iotest_mod, work_dir, rec_tl, monkeypatch):
    # read-only, read above target -> balance reverse until minimum -> accept
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 1000, "read": 10**9}, process_count=4)
    rc = _call_simul(iotest_mod, work_dir, rec_tl, None, 1000, process_count=4)
    assert rc == 0
    assert finalized


def test_simul_tuning_one_below_rebalances_ratio_then_exits(iotest_mod, work_dir, rec_tl, monkeypatch):
    # write below, read ok -> rebalance w/r until ratio fixed, then cannot
    # balance -> exit 1 (covers the ratio-adjust + "cannot balance" path)
    _stub_simul(iotest_mod, monkeypatch, {"write": 500, "read": 2000}, process_count=4)
    with pytest.raises(SystemExit) as ei:
        _call_simul(iotest_mod, work_dir, rec_tl, 1000, 2000, process_count=4)
    assert ei.value.code == 1


def test_simul_tuning_both_above_rebalances_and_accepts(iotest_mod, work_dir, rec_tl, monkeypatch):
    # Both roles well above target -> rebalance w/r then reach minimum config.
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 10**9, "read": 10**9}, process_count=4)
    # very low targets -> both measured above; search should still terminate
    rc = _call_simul(iotest_mod, work_dir, rec_tl, 1, 1, process_count=4)
    assert rc == 0
    assert finalized


def test_simul_tuning_asymmetric_above_ok_finalizes(iotest_mod, work_dir, rec_tl, monkeypatch):
    # write above / read ok asymmetric case -> ratio bump then finalize
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": 10**9, "read": 2000}, process_count=2)
    rc = _call_simul(iotest_mod, work_dir, rec_tl, 1000, 2000)
    assert rc == 0
    assert finalized


def test_simul_tuning_cannot_measure_exits(iotest_mod, work_dir, rec_tl, monkeypatch):
    finalized = _stub_simul(iotest_mod, monkeypatch, {"write": None, "read": None})
    with pytest.raises(SystemExit) as ei:
        _call_simul(iotest_mod, work_dir, rec_tl, 1000, 2000)
    assert ei.value.code == 1
