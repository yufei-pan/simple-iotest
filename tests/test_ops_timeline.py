import csv

import pytest


def read_ops(work_dir):
    (path,) = work_dir.glob("iotest_*_ops.csv")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_ops_timeline_has_one_row_per_op_and_matches_main_csv(run_main, work_dir, rec_tl):
    run_main(["write", "read"], process_count=2, file_count=3, tl=rec_tl, no_report=False)
    rows = read_ops(work_dir)
    assert list(rows[0]) == ["pass", "mode", "op", "worker", "file_index", "start_s", "duration_s", "error"]
    for pass_no, op in (("1", "write"), ("2", "read")):
        pass_rows = [r for r in rows if r["pass"] == pass_no]
        assert len(pass_rows) == 6
        assert {r["mode"] for r in pass_rows} == {op}
        assert {r["op"] for r in pass_rows} == {op}
        assert {r["worker"] for r in pass_rows} == {"0", "1"}
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


def test_ops_timeline_covers_comprehensive_and_simultaneous(run_main, work_dir, rec_tl):
    run_main(["comprehensive", "simultaneous"], process_count=2, file_count=2, tl=rec_tl, no_report=False)
    rows = read_ops(work_dir)
    comp = [r for r in rows if r["pass"] == "1"]
    assert sorted(r["op"] for r in comp) == sorted(["write", "move", "stat", "read"] * 4)
    simul = [r for r in rows if r["pass"] == "2"]
    assert {r["mode"] for r in simul} == {"simultaneous"}
    assert sorted(r["op"] for r in simul) == ["read"] * 2 + ["write"] * 2


def test_ops_timeline_records_failed_op_with_error_and_no_duration(run_main, work_dir, rec_tl):
    (work_dir / "0").mkdir()
    (work_dir / "0" / "temp_2.bin").mkdir()
    run_main(["write"], file_count=2, tl=rec_tl, no_report=False)
    rows = read_ops(work_dir)
    assert len(rows) == 2
    (failed,) = [r for r in rows if r["error"]]
    assert failed["file_index"] == "2"
    assert failed["duration_s"] == ""
    assert float(failed["start_s"]) >= 0
    assert [r["duration_s"] != "" for r in rows if not r["error"]] == [True]


def test_no_report_writes_no_ops_file(run_main, work_dir, rec_tl):
    run_main(["write"], tl=rec_tl, no_report=True)
    assert list(work_dir.glob("*_ops.csv")) == []


def test_write_timeline_orders_by_pass_then_start_and_formats_rows(iotest_mod, tmp_path):
    rows = [
        (1, "write", [("write", "1", 1, 0.5, 0.25, ""), ("write", "0", 1, 0.125, 0.5, "")]),
        (2, "read", [("read", "0", 1, 0.0, None, "boom"), ("read", "1", 2, 0.0, 1.5, "")]),
    ]
    path = tmp_path / "ops.csv"
    with path.open("w", newline="") as f:
        iotest_mod.write_timeline(rows)(f)
    assert path.read_bytes().decode().splitlines() == [
        "pass,mode,op,worker,file_index,start_s,duration_s,error",
        "1,write,write,0,1,0.125000,0.500000,",
        "1,write,write,1,1,0.500000,0.250000,",
        "2,read,read,0,1,0.000000,,boom",
        "2,read,read,1,2,0.000000,1.500000,",
    ]
