import csv
import os

import pytest


def test_meta_alias(iotest_mod):
    assert iotest_mod.parse_modes(["m"]) == ["meta"]


def test_meta_run_has_three_keys_iops_and_leaves_nothing(run_main, work_dir, rec_tl):
    result = run_main(["meta"], process_count=2, file_count=5, tl=rec_tl, no_report=False)
    for key in ("meta_create", "meta_stat", "meta_unlink"):
        assert len(result["outResults"][key]) == 10
        assert not result["failedOps"].get(key)
        assert result["totalTime"][key] > 0
    text = "\n".join(rec_tl.printed)
    assert "IOPS" in text
    assert "B/s" not in text
    # no meta files or directories remain
    assert not list(work_dir.rglob("meta"))
    assert not list(work_dir.rglob("meta_*"))
    (ops,) = work_dir.glob("iotest_*_ops.csv")
    with ops.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["mode"] for r in rows} == {"meta"}
    for pass_no, op in (("1", "create"), ("2", "stat"), ("3", "unlink")):
        pass_rows = [r for r in rows if r["pass"] == pass_no]
        assert len(pass_rows) == 10
        assert {r["op"] for r in pass_rows} == {op}


def test_meta_only_skips_generation_benchmark(run_main, rec_tl):
    run_main(["meta"], tl=rec_tl)
    assert any("Skipping file generation benchmark" in m for m in rec_tl.printed)


def test_meta_ignores_leftover_files(run_main, work_dir):
    meta = work_dir / "0" / "meta"
    meta.mkdir(parents=True)
    for i in (1, 2, 3):
        (meta / "meta_{}".format(i)).write_bytes(b"")
    result = run_main(["meta"], file_count=3)
    assert not any(result["failedOps"].values())
    assert not meta.exists()


def test_meta_file_missing_is_a_failed_op(iotest_mod, work_dir):
    path = str(work_dir / "nope")
    res = iotest_mod.meta_file("stat", path, quiet=True)
    assert res[0] is None and "FileNotFoundError" in res[1]
    assert iotest_mod.meta_file("unlink", path, quiet=True)[0] is None
    assert iotest_mod.meta_file("create", path, quiet=True)[0] is not None
    # O_EXCL: creating twice fails
    assert iotest_mod.meta_file("create", path, quiet=True)[0] is None
    os.unlink(path)


@pytest.mark.patches_workers
def test_meta_failed_stat_counts_in_run(run_main, iotest_mod, monkeypatch):
    real = iotest_mod.meta_file

    def flaky(phase, path, quiet=False):
        if phase == "stat" and path.endswith("meta_2"):
            return None, "FileNotFoundError: gone"
        return real(phase, path, quiet)

    monkeypatch.setattr(iotest_mod, "meta_file", flaky)
    result = run_main(["meta"], file_count=3)
    assert len(result["failedOps"]["meta_stat"]) == 1
    assert not result["failedOps"].get("meta_create")


def test_meta_cleanup_removes_leftover_meta_dirs(run_main, work_dir):
    (work_dir / "0" / "meta").mkdir(parents=True)
    run_main(["meta"], file_count=2, cleanup=True)
    assert not (work_dir / "0").exists()


def test_meta_reserves_no_memory(iotest_mod):
    assert iotest_mod.memory_copies(["meta"]) == 0
    assert iotest_mod.memory_copies(["write", "meta"]) == 1
