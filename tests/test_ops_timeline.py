import array
import csv
import inspect
import io
import json
import math
import subprocess
import sys

import pytest

from test_cli import MODULE


def read_ops(work_dir):
    (path,) = work_dir.glob("iotest_*_ops.csv")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_ops_timeline_has_one_row_per_op_and_matches_main_csv(run_main, work_dir, rec_tl):
    run_main(["write", "read"], process_count=2, file_count=3, tl=rec_tl, no_report=False, timeline=True)
    rows = read_ops(work_dir)
    assert list(rows[0]) == ["pass", "mode", "op", "worker", "file_index", "start_s", "duration_s", "error"]
    for pass_no, op in (("1", "write"), ("2", "read")):
        pass_rows = [r for r in rows if r["pass"] == pass_no]
        assert len(pass_rows) == 6
        assert {r["mode"] for r in pass_rows} == {op}
        assert {r["op"] for r in pass_rows} == {op}
        assert {r["worker"] for r in pass_rows} == {"0000", "0001"}
        assert sorted({r["file_index"] for r in pass_rows}) == ["1", "2", "3"]
        assert all(float(r["start_s"]) >= 0 and r["error"] == "" for r in pass_rows)
    # rows are sorted by pass, then start
    keys = [(int(r["pass"]), float(r["start_s"])) for r in rows]
    assert keys == sorted(keys)
    (main_csv,) = [p for p in work_dir.glob("iotest_*.csv") if not p.name.endswith("_ops.csv")]
    with main_csv.open(newline="") as f:
        main_rows = list(csv.DictReader(f))
    for i, op in enumerate(("write", "read")):
        from_ops = sorted(float(r["duration_s"]) for r in rows if r["pass"] == str(i + 1))
        from_main = sorted(float(r[op + "_time"]) for r in main_rows)
        assert from_ops == pytest.approx(from_main, abs=2e-6)
    assert any(m.startswith("Timeline written to ") for m in rec_tl.printed)


def test_ops_timeline_covers_comprehensive_simultaneous_and_meta(run_main, work_dir, rec_tl):
    run_main(["comprehensive", "simultaneous", "meta"], process_count=2, file_count=2, tl=rec_tl, no_report=False,
             timeline=True)
    rows = read_ops(work_dir)
    comp = [r for r in rows if r["pass"] == "1"]
    assert sorted(r["op"] for r in comp) == sorted(["write", "move", "stat", "read"] * 4)
    simul = [r for r in rows if r["pass"] == "2"]
    assert {r["mode"] for r in simul} == {"simultaneous"}
    assert sorted(r["op"] for r in simul) == ["read"] * 2 + ["write"] * 2
    for pass_no, op in (("3", "create"), ("4", "stat"), ("5", "unlink")):
        meta = [r for r in rows if r["pass"] == pass_no]
        assert {r["mode"] for r in meta} == {"meta"}
        assert [r["op"] for r in meta] == [op] * 4


def test_ops_timeline_records_failed_op_with_error_and_no_duration(run_main, work_dir, rec_tl):
    (work_dir / "0000").mkdir()
    (work_dir / "0000" / "temp_2.bin").mkdir()
    run_main(["write"], file_count=2, tl=rec_tl, no_report=False, timeline=True)
    rows = read_ops(work_dir)
    assert len(rows) == 2
    (failed,) = [r for r in rows if r["error"]]
    assert failed["file_index"] == "2"
    assert failed["duration_s"] == ""
    assert float(failed["start_s"]) >= 0
    assert [r["duration_s"] != "" for r in rows if not r["error"]] == [True]


@pytest.mark.parametrize("no_report", [False, True])
def test_timeline_is_off_by_default(run_main, work_dir, rec_tl, no_report):
    run_main(["write"], tl=rec_tl, no_report=no_report)
    assert list(work_dir.glob("*_ops.csv")) == []
    assert not any("Timeline" in m for m in rec_tl.messages)


@pytest.mark.parametrize("flags", [{"no_report": True}, {"stealth": True}])
def test_explicit_timeline_is_written_with_no_report_and_stealth(run_main, work_dir, rec_tl, flags):
    run_main(["write"], file_count=2, tl=rec_tl, timeline=True, **flags)
    # only the timeline: -nr and -S still skip the csv and the text report
    (ops,) = work_dir.glob("iotest_*")
    assert ops.name.endswith("_ops.csv")
    assert [r["file_index"] for r in read_ops(work_dir)] == ["1", "2"]


def test_cli_timeline_with_no_report(tmp_path):
    # a fresh interpreter with the platform's own start method: the records must reach the parent through the manager
    work = tmp_path / "w"
    proc = subprocess.run([sys.executable, str(MODULE), "-fs", "1k", "-fc", "2", "-pc", "2", "-d", str(work), "-nl", "-nr",
                           "-z", "-bt", "1", "-tl", "write"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Timeline written to " in proc.stdout
    assert len(read_ops(work)) == 4
    assert [p.name for p in work.glob("iotest_*") if not p.name.endswith("_ops.csv")] == []


def test_json_options_say_whether_the_timeline_was_on(run_main, rec_tl, tmp_path):
    path = tmp_path / "out.json"
    run_main(["write"], tl=rec_tl, json_path=str(path), timeline=True)
    assert json.loads(path.read_text())["options"]["timeline"] is True


def spy_worker_sinks(iotest_mod, monkeypatch):
    # the timeline argument each worker process is started with, by worker function
    sinks = []
    real = iotest_mod.Process

    def spy(*args, **kwargs):
        target = kwargs["target"]
        if target in (iotest_mod.worker, iotest_mod.simultaneous_role_worker):
            bound = inspect.signature(target).bind(*kwargs["args"])
            sinks.append((target.__name__, bound.arguments.get("timeline")))
        return real(*args, **kwargs)

    monkeypatch.setattr(iotest_mod, "Process", spy)
    return sinks


@pytest.mark.parametrize("timeline", [False, True])
def test_workers_get_a_timeline_sink_only_with_timeline(run_main, iotest_mod, monkeypatch, rec_tl, timeline):
    sinks = spy_worker_sinks(iotest_mod, monkeypatch)
    run_main(["write", "simultaneous", "meta"], process_count=2, tl=rec_tl, no_report=False, timeline=timeline)
    assert {name for name, _ in sinks} == {"worker", "simultaneous_role_worker"}
    # write 2 + simultaneous 1 + 1 + meta 3 x 2
    assert len(sinks) == 10
    assert all((sink is not None) == timeline for _, sink in sinks)


def test_tuning_attempts_record_no_timeline_but_the_final_run_does(iotest_mod, work_dir, rec_tl, monkeypatch):
    sinks = spy_worker_sinks(iotest_mod, monkeypatch)
    # pc=2 with a 1 B/s target: an attempt at 2, one at 1, then the final run at 1, which -nr does not skip
    # because the timeline asks for a file
    options = iotest_mod.RunOptions(file_size=1024, file_count=1, process_count=2, directories=str(work_dir),
        modes=["write"], zeros=True, no_report=True, benchmark_time=1, timeline=True)
    iotest_mod.tune_for_targets(options, rec_tl, 1, None)
    assert [sink is not None for _, sink in sinks] == [False, False, False, True]
    assert len(read_ops(work_dir)) == 1


def test_worker_without_a_sink_records_nothing(iotest_mod, work_dir, monkeypatch):
    def no_timeline(*args, **kwargs):
        raise AssertionError("a timeline was recorded without --timeline")

    monkeypatch.setattr(iotest_mod, "OpTimeline", no_timeline)
    results = []
    iotest_mod.worker(2, 1024, str(work_dir), results, "write", "0", True, True, 0)
    iotest_mod.simultaneous_role_worker("write", 2, 1024, str(work_dir), "t", "0", results, True, True, 0)
    # one results record each
    assert len(results) == 2
    assert len(list(iotest_mod.result_samples(results))) == 4


def unpack(record):
    counter, index_type, ops, file_indexes, starts, durations, errors = record
    # the arrays' bytes: 1 + 4 + 8 + 8 bytes an op
    assert [type(b) for b in (ops, file_indexes, starts, durations)] == [bytes] * 4
    assert index_type == "I"
    return (counter, list(ops), list(array.array("I", file_indexes)), list(array.array("d", starts)),
            list(array.array("d", durations)), errors)


def test_worker_hands_over_one_compact_record(iotest_mod, work_dir):
    sink = []
    iotest_mod.worker(3, 1024, str(work_dir), [], "comprehensive", "0", True, True, 0, timeline=sink)
    (record,) = sink
    counter, ops, file_indexes, starts, durations, errors = unpack(record)
    assert counter == "0" and errors == {}
    assert [iotest_mod.OP_NAMES[code] for code in ops] == ["write", "move", "stat", "read"] * 3
    assert file_indexes == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert starts == sorted(starts)
    assert all(d >= 0 for d in durations)


def test_op_timeline_keeps_failed_ops_apart(iotest_mod):
    timeline = iotest_mod.OpTimeline("3", 10.0, 2)
    timeline.add("read", 1, 10.0, (10.5, 10.75))
    timeline.add("read", 2, 11.0, (None, "OSError: boom"))
    counter, ops, file_indexes, starts, durations, errors = unpack(timeline.record())
    assert counter == "3" and file_indexes == [1, 2]
    assert starts == [0.5, 1.0]
    assert durations[0] == 0.25 and math.isnan(durations[1])
    assert errors == {1: "OSError: boom"}


def test_op_timeline_takes_file_indexes_past_32_bits(iotest_mod):
    timeline = iotest_mod.OpTimeline("0", 0.0, 2 ** 32)
    timeline.add("index", 2 ** 32, 0.0, (0.5, 0.75))
    out = io.StringIO()
    iotest_mod.write_timeline([(1, "index", [timeline.record()])])(out)
    assert out.getvalue().splitlines()[1] == "1,index,index,0,4294967296,0.500000,0.250000,"


def synthetic_passes(iotest_mod):
    # pass 1: worker 1's record arrived first. It has a failed op at 0.75, the same start as
    # worker 0's second op, so that tie keeps worker 1's row first, as the old stable sort did
    w1 = iotest_mod.OpTimeline("1", 10.0, 3)
    w1.add("write", 1, 10.0, (10.5, 10.75))
    w1.add("write", 2, 10.75, (None, "OSError: [Errno 28] No space left on device, 'x'"))
    w1.add("write", 3, 11.0, (11.0, 11.5))
    w0 = iotest_mod.OpTimeline("0", 10.0, 3)
    w0.add("write", 1, 10.0, (10.125, 10.625))
    w0.add("write", 2, 10.625, (10.75, 11.0))
    # pass 2: meta ops, a worker with no ops at all, and a tie at 0.0 between two workers
    m0 = iotest_mod.OpTimeline("0", 20.0, 2)
    m0.add("create", 1, 20.0, (20.0, 20.0625))
    m0.add("create", 2, 20.0625, (20.0625, 20.125))
    empty = iotest_mod.OpTimeline("2", 20.0, 2)
    m1 = iotest_mod.OpTimeline("1", 20.0, 2)
    m1.add("create", 1, 20.0, (20.0, 20.5))
    return [(1, "write", [w1.record(), w0.record()]), (2, "meta", [m0.record(), empty.record(), m1.record()])]


EXPECTED_TIMELINE = (
    "pass,mode,op,worker,file_index,start_s,duration_s,error\r\n"
    "1,write,write,0,1,0.125000,0.500000,\r\n"
    "1,write,write,1,1,0.500000,0.250000,\r\n"
    "1,write,write,1,2,0.750000,,\"OSError: [Errno 28] No space left on device, 'x'\"\r\n"
    "1,write,write,0,2,0.750000,0.250000,\r\n"
    "1,write,write,1,3,1.000000,0.500000,\r\n"
    "2,meta,create,0,1,0.000000,0.062500,\r\n"
    "2,meta,create,1,1,0.000000,0.500000,\r\n"
    "2,meta,create,0,2,0.062500,0.062500,\r\n"
)


def test_write_timeline_merges_compact_records_into_the_old_format(iotest_mod):
    out = io.StringIO()
    iotest_mod.write_timeline(synthetic_passes(iotest_mod))(out)
    assert out.getvalue() == EXPECTED_TIMELINE


def test_collect_timeline_fetches_once_and_empties_the_manager_list(iotest_mod):
    class FakeProxy(list):
        fetches = 0

        def __getitem__(self, index):
            FakeProxy.fetches += 1
            return list.__getitem__(self, index)

    proxy = FakeProxy(["a", "b"])
    passes = []
    iotest_mod.collect_timeline(passes, proxy, 1, "write")
    assert passes == [(1, "write", ["a", "b"])]
    assert FakeProxy.fetches == 1
    assert proxy == []
    # without --timeline there is no list to fetch
    iotest_mod.collect_timeline(None, None, 2, "write")
