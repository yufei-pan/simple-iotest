import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "iotest.py"


def blocks(data, size=512):
    return {data[o:o + size] for o in range(0, len(data), size)}


def worker_files(work_dir):
    return [(work_dir / "0000" / name).read_bytes() for name in ("temp_1.bin", "temp_2.bin")]


def test_files_in_a_worker_are_identical_by_default(run_main, work_dir):
    # stamping costs sync time, so it is opt-in; the default keeps old reports comparable
    run_main(["write"], file_count=2, file_size=8192, zeros=False)
    first, second = worker_files(work_dir)
    assert first == second


def test_unique_files_share_no_block(run_main, work_dir):
    run_main(["write"], file_count=2, file_size=8192, zeros=False, unique=True)
    first, second = worker_files(work_dir)
    assert len(first) == len(second) == 8192
    # identical blocks would let dedup storage keep one physical copy for both files
    assert blocks(first).isdisjoint(blocks(second))


def test_unique_simultaneous_writer_and_layout_files_share_no_block(run_main, work_dir):
    run_main(["simultaneous"], process_count=2, file_count=2, file_size=8192, zeros=False, unique=True,
        simultaneous_run_id="t", simultaneous_skip_cleanup=True)
    worker_dir = work_dir / ".iotest_simul_t" / "worker_0000"
    for a, b in (("write_1.bin", "write_2.bin"), ("read_1.bin", "read_2.bin")):
        assert blocks((worker_dir / a).read_bytes()).isdisjoint(blocks((worker_dir / b).read_bytes()))


def test_zeros_files_stay_all_zero_by_default(run_main, work_dir):
    run_main(["write"], file_count=2, file_size=8192, zeros=True)
    assert worker_files(work_dir) == [b"\x00" * 8192, b"\x00" * 8192]


def test_unique_reaches_tuning_runs(iotest_mod, work_dir, rec_tl, tiny):
    # pc=1 with a 1 B/s target: one tuning attempt, accepted as best effort
    options = iotest_mod.RunOptions(file_size=8192, file_count=2, process_count=1, directories=str(work_dir),
        modes=["write"], quiet=True, zeros=False, no_report=True, benchmark_time=tiny["benchmarkTime"], unique=True)
    iotest_mod.tune_for_targets(options, rec_tl, 1, None)
    first, second = worker_files(work_dir)
    assert blocks(first).isdisjoint(blocks(second))


def test_cli_unique_flag(tmp_path):
    work = tmp_path / "w"
    proc = subprocess.run(
        [sys.executable, str(MODULE), "-fs", "8k", "-fc", "2", "-pc", "1", "-d", str(work),
         "-nl", "-nr", "-bt", "1", "-u", "write"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    first, second = worker_files(work)
    assert blocks(first).isdisjoint(blocks(second))


def test_stamp_unique_handles_tail_shorter_than_the_tag(iotest_mod):
    # 1030 bytes leaves a 6-byte final block, shorter than the 24-byte tag
    buf = bytearray(os.urandom(1030))
    other = bytearray(buf)
    iotest_mod.stamp_unique(buf, 1, 0)
    iotest_mod.stamp_unique(other, 1, 1)
    assert len(buf) == len(other) == 1030
    assert blocks(bytes(buf)).isdisjoint(blocks(bytes(other)))


def test_zeros_with_unique_has_no_repeated_block(run_main, work_dir):
    # with -z every block is zeros apart from the tag, so the tag itself must differ per block
    run_main(["write"], file_count=2, file_size=8192, zeros=True, unique=True)
    first, second = worker_files(work_dir)
    assert len(blocks(first)) == len(blocks(second)) == 16
    assert blocks(first).isdisjoint(blocks(second))


def stamp(iotest_mod, use_numpy, size):
    iotest_mod.numpy_available = use_numpy
    buf = bytearray(size)
    iotest_mod.stamp_unique(buf, 4321, 7)
    return bytes(buf)


def test_stamp_unique_blocks_and_tail_are_all_distinct(iotest_mod, monkeypatch):
    monkeypatch.setattr(iotest_mod, "numpy_available", iotest_mod.numpy_available)
    data = stamp(iotest_mod, False, 8 * 512 + 100)
    assert len(blocks(data)) == 9


def test_numpy_and_pure_python_stamps_are_identical(iotest_mod, monkeypatch):
    if not iotest_mod.numpy_available:
        pytest.skip("numpy not installed")
    monkeypatch.setattr(iotest_mod, "numpy_available", True)
    for size in (8 * 512, 8 * 512 + 10, 8 * 512 + 100, 300):
        assert stamp(iotest_mod, True, size) == stamp(iotest_mod, False, size)


def test_stamps_do_not_repeat_when_a_pid_is_reused(iotest_mod, tmp_path):
    # both calls run in this test process, so they share a PID, as two runs can after
    # PID reuse; kept files from an earlier run must still not match the new run's blocks
    dirs = []
    for name in ("run1", "run2"):
        d = tmp_path / name
        d.mkdir()
        iotest_mod.worker(1, 4096, str(d), [], "write", "0", True, True, 0, True)
        dirs.append((d / "0" / "temp_1.bin").read_bytes())
    assert blocks(dirs[0]).isdisjoint(blocks(dirs[1]))


def test_unique_stamps_files_that_read_mode_recreates(run_main, work_dir):
    # read on a fresh directory recreates every file before reading it
    run_main(["read"], file_count=2, file_size=8192, zeros=True, unique=True)
    first, second = worker_files(work_dir)
    assert len(blocks(first)) == len(blocks(second)) == 16
    assert blocks(first).isdisjoint(blocks(second))


def test_unique_read_checks_each_file_once(iotest_mod, work_dir, monkeypatch):
    # the check runs inside the measured wall-clock window; read_file already does it
    iotest_mod.worker(3, 4096, str(work_dir), [], "write", "0", True, True, 0)
    calls = []
    real = iotest_mod.file_needs_recreate
    monkeypatch.setattr(iotest_mod, "file_needs_recreate", lambda *a: calls.append(a) or real(*a))
    iotest_mod.worker(3, 4096, str(work_dir), [], "read", "0", True, True, 0, True)
    assert len(calls) == 3
