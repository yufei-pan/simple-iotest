import math

import pytest


def block_with_dir(path):
    # a directory where iotest expects a regular file makes open() fail
    path.mkdir(parents=True)
    return path


@pytest.mark.parametrize("helper", ["create_file", "read_file"])
def test_data_helpers_return_error_text_on_failure(iotest_mod, tmp_path, helper):
    blocker = block_with_dir(tmp_path / "f.bin")
    op = iotest_mod.op_time(getattr(iotest_mod, helper)(str(blocker), b"\x00" * 64, 64, quiet=True))
    assert isinstance(op, str)
    assert str(blocker) in op


def test_metadata_helpers_return_error_text_on_failure(iotest_mod, tmp_path):
    missing = tmp_path / "missing.bin"
    for result in (
        iotest_mod.move_file(str(missing), str(tmp_path / "dst.bin"), quiet=True),
        iotest_mod.stat_file(str(missing), quiet=True),
    ):
        op = iotest_mod.op_time(result)
        assert isinstance(op, str)
        assert str(missing) in op
    # a regular file where index_file wants to create its directory
    (tmp_path / "x.bin.index").write_bytes(b"")
    op = iotest_mod.op_time(iotest_mod.index_file(str(tmp_path / "x.bin"), quiet=True))
    assert isinstance(op, str)
    assert "x.bin.index" in op


def test_failed_write_is_counted_not_timed(run_main, work_dir, rec_tl):
    blocker = block_with_dir(work_dir / "0" / "temp_2.bin")
    result = run_main(["write"], file_count=2, tl=rec_tl)
    times = result["outResults"]["write"]
    assert len(times) == 1
    # a failure used to be recorded as seconds-since-boot
    assert times[0] < 60
    errors = result["failedOps"]["write"]
    assert len(errors) == 1
    assert str(blocker) in errors[0]
    # only the one successful 1 KiB write counts toward the reported size
    size_line = next(line for line in "\n".join(rec_tl.printed).splitlines() if line.startswith("Total write size"))
    assert size_line.endswith("1.00 KiB")


def test_all_ops_failed_reports_first_error_without_crashing(run_main, work_dir, rec_tl):
    blocker = block_with_dir(work_dir / "0" / "temp_1.bin")
    result = run_main(["write"], tl=rec_tl)
    assert result["outResults"]["write"] == []
    assert len(result["failedOps"]["write"]) == 1
    assert str(blocker) in "\n".join(rec_tl.printed)


def test_comprehensive_failures_stay_on_their_own_op(run_main, work_dir):
    block_with_dir(work_dir / "0" / "temp_1.bin")
    result = run_main(["comprehensive"])
    failed = {key: len(errors) for key, errors in result["failedOps"].items()}
    # the write fails on the blocker directory; rename and stat work on a directory,
    # then reading the renamed directory fails
    assert failed == {"write": 1, "move": 0, "stat": 0, "read": 1}
    assert all(math.isfinite(t) for t in result["totalTime"].values())


def test_simultaneous_write_failure_is_counted(run_main, work_dir):
    block_with_dir(work_dir / ".iotest_simul_t" / "worker_0" / "write_1.bin")
    result = run_main(["simultaneous"], process_count=2, simultaneous_run_id="t")
    assert result["outResults"]["write"] == []
    assert len(result["failedOps"]["write"]) == 1
    assert result["failedOps"]["read"] == []
    assert len(result["outResults"]["read"]) == 1


def test_failed_ops_keys_follow_repeated_mode_renames(run_main, work_dir):
    block_with_dir(work_dir / "0" / "temp_1.bin")
    result = run_main(["write", "write"])
    assert set(result["failedOps"]) == {"write_1", "write_2"}
    assert set(result["failedOps"]) == set(result["outResults"])


@pytest.mark.parametrize(
    "blocked,file_count,expected",
    [
        ("temp_2.bin", 2, "1 of 2 ops failed"),
        ("temp_1.bin", 1, "all 1 ops failed"),
    ],
)
def test_short_report_mentions_failures(run_main, work_dir, webhook, blocked, file_count, expected):
    url, received = webhook()
    block_with_dir(work_dir / "0" / blocked)
    run_main(["write"], file_count=file_count, message_end_point_address=url)
    (request,) = received()
    assert expected in request["json"]["content"]


def test_tuning_attempt_with_failures_exits(iotest_mod, work_dir, rec_tl, tiny):
    block_with_dir(work_dir / "0" / "temp_1.bin")
    options = iotest_mod.RunOptions(
        file_size=tiny["file_size"], file_count=1, process_count=1, directories=str(work_dir),
        modes=["write"], quiet=True, zeros=True, stealth=False, threshold_to_report_anomaly=0,
        benchmark_time=tiny["benchmarkTime"])
    with pytest.raises(SystemExit) as ei:
        iotest_mod._run_tuning_main(options, rec_tl, 1)
    assert ei.value.code == 1


def test_final_tuning_run_with_failures_exits(iotest_mod, work_dir, rec_tl, tiny):
    block_with_dir(work_dir / "0" / "temp_1.bin")
    # no_report=False is what makes the final run happen at all
    options = iotest_mod.RunOptions(
        file_size=tiny["file_size"], file_count=1, process_count=1, directories=str(work_dir),
        modes=["write"], quiet=True, zeros=True, stealth=False, message_end_point_address=None,
        no_report=False, threshold_to_report_anomaly=0, benchmark_time=tiny["benchmarkTime"])
    with pytest.raises(SystemExit) as ei:
        iotest_mod._finalize_tuning_run(options, rec_tl, 1)
    assert ei.value.code == 1


def test_worker_that_dies_counts_its_ops_as_failed(run_main, work_dir, webhook):
    # a regular file where the worker wants its directory: os.makedirs raises and the
    # process dies before its loop, handing back no results at all
    (work_dir / "0").write_text("")
    url, received = webhook()
    result = run_main(["write"], file_count=3, message_end_point_address=url)
    assert result["outResults"]["write"] == []
    assert len(result["failedOps"]["write"]) == 3
    (request,) = received()
    assert "all 3 ops failed" in request["json"]["content"]


def test_dead_comprehensive_worker_fails_every_step(run_main, work_dir):
    (work_dir / "0").write_text("")
    result = run_main(["comprehensive"], file_count=2)
    assert {key: len(errors) for key, errors in result["failedOps"].items()} == {
        "write": 2, "move": 2, "stat": 2, "read": 2}


def test_failed_recreate_reports_the_write_error(iotest_mod, tmp_path, monkeypatch):
    # read_file recreates a missing file first; if that write fails, its error is the real cause
    monkeypatch.setattr(iotest_mod, "create_file",
        lambda *a, **k: (None, "OSError: [Errno 28] No space left on device"))
    op = iotest_mod.op_time(iotest_mod.read_file(str(tmp_path / "missing.bin"), b"\x00" * 64, 64, quiet=True))
    assert isinstance(op, str)
    assert "No space left on device" in op
