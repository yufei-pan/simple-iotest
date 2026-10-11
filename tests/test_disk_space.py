import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "iotest.py"


def fake_free(iotest_mod, monkeypatch, free):
    monkeypatch.setattr(iotest_mod.shutil, "disk_usage",
        lambda path: types.SimpleNamespace(total=free, used=0, free=free))


def temp_files(work_dir):
    return sorted(p.relative_to(work_dir).as_posix() for p in Path(work_dir).rglob("temp_*.bin"))


def test_refuses_before_writing_when_space_is_short(run_main, work_dir, iotest_mod, monkeypatch, rec_tl):
    fake_free(iotest_mod, monkeypatch, 1000)
    with pytest.raises(SystemExit) as ei:
        run_main(["write"], tl=rec_tl)  # needs 1 x 1024 B
    assert ei.value.code == 1
    assert temp_files(work_dir) == []
    assert any(str(work_dir) in m for m in rec_tl.printed)


def test_files_from_a_previous_run_are_reclaimable(run_main, work_dir, iotest_mod, monkeypatch):
    run_main(["write"])
    fake_free(iotest_mod, monkeypatch, 0)
    # rewriting temp_1.bin frees its old 1024 B first, so no new space is needed
    result = run_main(["write"])
    assert len(result["outResults"]["write"]) == 1


@pytest.mark.parametrize(
    "modes,free,refuses",
    [
        # fs=1024 fc=10 pc=1: write keeps 10 files, comprehensive deletes each one after reading it
        (["write"], 2000, True),
        (["comprehensive"], 2000, False),
        (["index"], 0, False),
        # simultaneous pc=2 fc=2: 2 layout workers x 2 files + 1 writer x 2 files = 6 x 1024 = 6144 B
        (["simultaneous"], 6000, True),
        (["simultaneous"], 7000, False),
    ],
)
def test_space_needed_per_mode(run_main, work_dir, iotest_mod, monkeypatch, modes, free, refuses):
    fake_free(iotest_mod, monkeypatch, free)
    params = {"process_count": 2, "file_count": 2} if modes == ["simultaneous"] else {"file_count": 10}
    if refuses:
        with pytest.raises(SystemExit):
            run_main(modes, **params)
    else:
        run_main(modes, **params)


def test_cleanup_removes_worker_files_and_empty_dirs(run_main, work_dir):
    run_main(["write", "read"], process_count=2, file_count=2, cleanup=True)
    assert list(Path(work_dir).iterdir()) == []


def test_cleanup_keeps_files_it_did_not_create(run_main, work_dir):
    keep = work_dir / "0" / "notes.txt"
    keep.parent.mkdir()
    keep.write_text("mine")
    run_main(["write"], file_count=2, cleanup=True)
    assert temp_files(work_dir) == []
    assert keep.read_text() == "mine"


def run_cli(*args):
    return subprocess.run([sys.executable, str(MODULE), *args], capture_output=True, text=True, check=False)


def test_without_cleanup_files_stay_and_a_hint_is_printed(tmp_path):
    work = tmp_path / "w"
    proc = run_cli("-fs", "1k", "-fc", "2", "-pc", "2", "-d", str(work), "-nl", "-nr", "-z", "-bt", "1", "write")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert temp_files(work) == ["0/temp_1.bin", "0/temp_2.bin", "1/temp_1.bin", "1/temp_2.bin"]
    assert "--cleanup" in proc.stdout


def test_tuning_with_cleanup_prints_no_cleanup_hint(tmp_path):
    # the final tuning run used to suggest -cl although -cl was given
    work = tmp_path / "w"
    proc = run_cli("-fs", "1k", "-fc", "2", "-pc", "1", "-d", str(work), "-nl", "-z", "-bt", "1",
                   "-wtpt", "1k", "-cl", "write")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert temp_files(work) == []
    assert "--cleanup" not in proc.stdout


def test_leftover_hint_counts_files_from_every_counter_padding(iotest_mod, work_dir):
    for pc in (12, 3):
        for _, _, path in iotest_mod.worker_files([str(work_dir)], pc, 1):
            Path(path).parent.mkdir(exist_ok=True)
            Path(path).write_bytes(b"x")
    assert "Left 15 test files" in iotest_mod.leftover_hint([str(work_dir)], 12, 1)


def test_cleanup_reaches_tuning_runs(iotest_mod, work_dir, rec_tl, tiny):
    iotest_mod.tune_for_targets(1024, 2, 1, str(work_dir), ["write"], True, True, rec_tl,
        1, None, no_report=True, benchmarkTime=tiny["benchmarkTime"], cleanup=True)
    assert temp_files(work_dir) == []


def test_cli_cleanup_flag(tmp_path):
    work = tmp_path / "w"
    proc = subprocess.run(
        [sys.executable, str(MODULE), "-fs", "1k", "-fc", "2", "-pc", "1", "-d", str(work),
         "-nl", "-nr", "-z", "-bt", "1", "-cl", "write"],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert list(work.iterdir()) == []


def test_read_counts_the_files_it_must_create(run_main, work_dir, iotest_mod, monkeypatch):
    # read_file recreates a missing file at full size before reading it
    fake_free(iotest_mod, monkeypatch, 1000)
    with pytest.raises(SystemExit):
        run_main(["read"])
    assert temp_files(work_dir) == []


def test_read_of_existing_files_needs_no_space(run_main, work_dir, iotest_mod, monkeypatch):
    run_main(["write"])
    fake_free(iotest_mod, monkeypatch, 0)
    result = run_main(["read"])
    assert len(result["outResults"]["read"]) == 1


def test_simultaneous_tuning_checks_space_before_its_layout(iotest_mod, work_dir, rec_tl, monkeypatch, tiny):
    layouts = []
    monkeypatch.setattr(iotest_mod, "run_simultaneous_layout", lambda *a, **k: layouts.append(a))
    fake_free(iotest_mod, monkeypatch, 1000)
    with pytest.raises(SystemExit) as ei:
        iotest_mod.tune_for_targets(1024, 2, 2, str(work_dir), ["simultaneous"], True, True, rec_tl,
            1, None, no_report=True, benchmarkTime=tiny["benchmarkTime"])
    assert ei.value.code == 1
    assert layouts == []


def test_tuning_cleans_up_only_after_the_last_attempt(run_main, iotest_mod, work_dir, rec_tl, monkeypatch, tiny):
    # deleting files between attempts would make the next read attempt recreate them inside its timing
    run_main(["write"], process_count=2, file_count=2)
    events = []
    real_attempt, real_cleanup = iotest_mod._run_tuning_main, iotest_mod.cleanup_worker_files
    monkeypatch.setattr(iotest_mod, "_run_tuning_main",
        lambda *a, **k: events.append("attempt") or real_attempt(*a, **k))
    monkeypatch.setattr(iotest_mod, "cleanup_worker_files",
        lambda *a, **k: events.append("cleanup") or real_cleanup(*a, **k))
    # a 1 B/s read target: pc 2 is above it, so pc drops to 1 and the second attempt is accepted
    iotest_mod.tune_for_targets(1024, 2, 2, str(work_dir), ["read"], True, True, rec_tl,
        None, 1, no_report=True, benchmarkTime=tiny["benchmarkTime"], cleanup=True)
    assert events.count("attempt") == 2
    last_attempt = max(i for i, e in enumerate(events) if e == "attempt")
    assert "cleanup" not in events[:last_attempt]
    assert temp_files(work_dir) == []


def test_cleanup_after_tuning_covers_every_counter_padding(iotest_mod, work_dir):
    # attempts at -pc 12 name worker dirs 00..11, attempts at -pc 3 name them 0..2
    for pc in (12, 3):
        for _, _, path in iotest_mod.worker_files([str(work_dir)], pc, 1):
            Path(path).parent.mkdir(exist_ok=True)
            Path(path).write_bytes(b"x")
    iotest_mod.cleanup_all_worker_files([str(work_dir)], 12, 1)
    assert list(Path(work_dir).iterdir()) == []


def mountinfo_line(mount_point, fstype, source):
    return f"36 1 0:50 / {mount_point} rw,relatime shared:1 - {fstype} {source} rw"


ROOT = mountinfo_line("/", "ext4", "/dev/sda1")


@pytest.mark.parametrize(
    "a,b,same",
    [
        # btrfs subvolumes get their own st_dev but share the filesystem's free space
        (("/pool/a", "btrfs", "/dev/sdb"), ("/pool/b", "btrfs", "/dev/sdb"), True),
        (("/pool/a", "btrfs", "/dev/sdb"), ("/other", "btrfs", "/dev/sdc"), False),
        # ZFS datasets share their pool's free space
        (("/tank/a", "zfs", "tank/a"), ("/tank/b", "zfs", "tank/b"), True),
        (("/tank/a", "zfs", "tank/a"), ("/vault/b", "zfs", "vault/b"), False),
    ],
)
def test_space_pool(iotest_mod, a, b, same):
    lines = [ROOT, mountinfo_line(*a), mountinfo_line(*b)]
    assert (iotest_mod.space_pool(a[0], lines) == iotest_mod.space_pool(b[0], lines)) == same


def test_free_space_is_checked_per_pool(run_main, tmp_path, iotest_mod, monkeypatch):
    # two directories on different btrfs filesystems: each has its own free space
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    lines = [ROOT, mountinfo_line(str(d1), "btrfs", "/dev/sdb"), mountinfo_line(str(d2), "btrfs", "/dev/sdc")]
    monkeypatch.setattr(iotest_mod, "read_mountinfo", lambda: lines)
    fake_free(iotest_mod, monkeypatch, 1500)
    # 2 workers round-robin: 1024 B to each directory, which fits each pool on its own
    result = run_main(["write"], process_count=2, directory=[str(d1), str(d2)])
    assert len(result["outResults"]["write"]) == 2


def test_disk_check_comes_before_the_memory_pause(run_main, iotest_mod, monkeypatch):
    # little RAM but plenty of swap: no process cut, but the 90% warning pauses 16 s
    sleeps = []
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (1000, 10**9))
    fake_free(iotest_mod, monkeypatch, 0)
    with pytest.raises(SystemExit):
        run_main(["write"])
    assert 16 not in sleeps


def test_simultaneous_attempts_rely_on_the_tuners_upfront_check(iotest_mod, work_dir, rec_tl, monkeypatch):
    # attempts overwrite the writer files of earlier attempts, which still hold their space;
    # counting them again as new could stop tuning half way with a false "not enough space"
    iotest_mod.run_simultaneous_layout(1, 1024, [str(work_dir)], "r", 1, 2, True, True, rec_tl)
    fake_free(iotest_mod, monkeypatch, 0)
    result = iotest_mod.main(1024, 1, 2, str(work_dir), ["simultaneous"], True, True, tl=rec_tl,
        no_report=True, benchmarkTime=1, simultaneous_run_id="r", simultaneous_skip_layout=True,
        simultaneous_layout_process_count=2)
    assert len(result["outResults"]["write"]) == 1


def test_rewrites_on_copy_on_write_filesystems_are_not_reclaimable(run_main, work_dir, iotest_mod, monkeypatch):
    # on btrfs/ZFS a snapshot can keep the old blocks of a rewritten file allocated
    run_main(["write"])
    lines = [ROOT, mountinfo_line(os.path.realpath(work_dir), "btrfs", "/dev/sdb")]
    monkeypatch.setattr(iotest_mod, "read_mountinfo", lambda: lines)
    fake_free(iotest_mod, monkeypatch, 0)
    with pytest.raises(SystemExit):
        run_main(["write"])
    # reading existing full-size files rewrites nothing, so it still needs no space
    result = run_main(["read"])
    assert len(result["outResults"]["read"]) == 1


def test_each_directory_is_checked_against_its_own_free_space(run_main, tmp_path, iotest_mod, monkeypatch):
    # same btrfs pool, but d1 is limited (qgroup / ZFS quota): it must not borrow d2's room
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    lines = [ROOT, mountinfo_line(str(d1), "btrfs", "/dev/sdb"), mountinfo_line(str(d2), "btrfs", "/dev/sdb")]
    monkeypatch.setattr(iotest_mod, "read_mountinfo", lambda: lines)
    free = {str(d1): 10, str(d2): 10**9}
    monkeypatch.setattr(iotest_mod.shutil, "disk_usage",
        lambda path: types.SimpleNamespace(total=free[str(path)], used=0, free=free[str(path)]))
    with pytest.raises(SystemExit):
        run_main(["write"], process_count=2, directory=[str(d2), str(d1)])
