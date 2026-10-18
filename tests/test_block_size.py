import functools
import json
import os
import tracemalloc

import pytest

from test_cli import run_iotest

KiB = 2**10
MiB = 2**20
REAL_WRITEV = os.writev
REAL_READV = os.readv


def blocks(data, size=512):
    return [data[o:o + size] for o in range(0, len(data), size)]


def all_unique(*files):
    every = [block for data in files for block in blocks(data)]
    return len(set(every)) == len(every)


def stamp_paths(iotest_mod):
    yield False
    if iotest_mod.numpy_available:
        yield True


def worker_files(work_dir, count=2):
    return [(work_dir / "0" / f"temp_{i}.bin").read_bytes() for i in range(1, count + 1)]


def failures(results):
    # random results are (kind, time), comprehensive ones [write, move, stat, read]
    flat = []
    for r in results:
        flat.extend(r[1:] if isinstance(r, tuple) else r if isinstance(r, list) else [r])
    return [r for r in flat if isinstance(r, str)]


def traced_peak(fn, *args, **kwargs):
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("zeros", [True, False])
@pytest.mark.parametrize("block_size", [1024, 1000])
def test_chunked_unique_files_keep_their_size_and_never_repeat_a_block(iotest_mod, tmp_path, monkeypatch, zeros, block_size):
    # 1000 is not a multiple of 512, so stamps straddle the chunks
    for use_numpy in stamp_paths(iotest_mod):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        d = tmp_path / str(use_numpy)
        d.mkdir()
        iotest_mod.worker(2, 8192 + 100, str(d), [], "write", "0", True, zeros, 0, True, block_size=block_size)
        files = worker_files(d)
        assert [len(data) for data in files] == [8292, 8292]
        assert all_unique(*files)


@pytest.mark.parametrize("block_size", [512, 1000, 100, 7])
def test_chunk_stamps_land_where_stamping_the_whole_file_puts_them(iotest_mod, tmp_path, monkeypatch, block_size):
    size = 8 * 512 + 100
    tag = iotest_mod.UNIQUE_TAG.size
    for use_numpy in stamp_paths(iotest_mod):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        whole = bytearray(size)
        iotest_mod.stamp_unique(whole, 4321, 7)
        target = tmp_path / "f.bin"
        stamp = functools.partial(iotest_mod.stamp_chunk, worker_id=4321, file_index=7)
        op = iotest_mod.op_time(iotest_mod.create_file(str(target), bytearray(block_size), size, quiet=True,
            block_size=block_size, stamp=stamp))
        assert not isinstance(op, str), op
        data = target.read_bytes()
        assert len(data) == size
        for start in range(0, size, 512):
            assert data[start:start + tag] == bytes(whole[start:start + tag]), start


def test_stamp_unique_numbers_blocks_from_first_block(iotest_mod, monkeypatch):
    for use_numpy in stamp_paths(iotest_mod):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        whole = bytearray(4 * 512 + 100)
        iotest_mod.stamp_unique(whole, 9, 3)
        part = bytearray(2 * 512 + 100)
        iotest_mod.stamp_unique(memoryview(part), 9, 3, first_block=2)
        assert bytes(part) == bytes(whole[2 * 512:])


def test_without_unique_a_chunked_file_repeats_its_block(run_main, work_dir):
    # what the README warns about: deduplicating storage may keep one copy of the block
    run_main(["write"], file_size=8192, zeros=False, block_size=1024)
    data = worker_files(work_dir, 1)[0]
    assert len(data) == 8192
    assert data == data[:1024] * 8


def spy_calls(monkeypatch, iotest_mod, name, real, sizes):
    def spy(fd, buffers):
        sizes.append(len(buffers[0]))
        return real(fd, buffers)
    monkeypatch.setattr(iotest_mod.os, name, spy)


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_create_file_writes_block_sized_calls(iotest_mod, tmp_path, monkeypatch):
    block = os.urandom(1000)
    target = tmp_path / "f.bin"
    sizes = []
    spy_calls(monkeypatch, iotest_mod, "writev", REAL_WRITEV, sizes)
    op = iotest_mod.op_time(iotest_mod.create_file(str(target), block, 2500, quiet=True, block_size=1000))
    assert not isinstance(op, str), op
    assert sizes == [1000, 1000, 500]
    assert target.read_bytes() == block * 2 + block[:500]


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_read_file_reads_the_whole_file_in_block_sized_calls(iotest_mod, tmp_path, monkeypatch):
    content = os.urandom(2500)
    target = tmp_path / "f.bin"
    target.write_bytes(content)
    sizes = []
    spy_calls(monkeypatch, iotest_mod, "readv", REAL_READV, sizes)
    buffer = bytearray(1000)
    op = iotest_mod.op_time(iotest_mod.read_file(str(target), None, 2500, quiet=True, buffer=buffer, block_size=1000))
    assert not isinstance(op, str), op
    assert sizes == [1000, 1000, 500]
    assert bytes(buffer[:500]) == content[2000:]


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_blocks_larger_than_max_io_chunk_are_still_split(iotest_mod, tmp_path, monkeypatch):
    content = os.urandom(1000)
    target = tmp_path / "f.bin"
    sizes = []
    monkeypatch.setattr(iotest_mod, "MAX_IO_CHUNK", 512)
    spy_calls(monkeypatch, iotest_mod, "writev", REAL_WRITEV, sizes)
    spy_calls(monkeypatch, iotest_mod, "readv", REAL_READV, sizes)
    iotest_mod.create_file(str(target), content, 2500, quiet=True, block_size=1000)
    iotest_mod.read_file(str(target), None, 2500, quiet=True, block_size=1000)
    assert sizes == [512, 488, 512, 488, 500] * 2


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_chunked_io_finishes_short_transfers(iotest_mod, tmp_path, monkeypatch):
    # like tests/test_short_io.py: each call moves at most 300 bytes, less than a block
    def short_writev(fd, buffers):
        return os.write(fd, bytes(buffers[0][:300]))

    def short_readv(fd, buffers):
        data = os.read(fd, min(len(buffers[0]), 300))
        buffers[0][:len(data)] = data
        consumed.append(len(data))
        return len(data)

    consumed = []
    block = os.urandom(1000)
    target = tmp_path / "f.bin"
    monkeypatch.setattr(iotest_mod.os, "writev", short_writev)
    monkeypatch.setattr(iotest_mod.os, "readv", short_readv)
    write = iotest_mod.op_time(iotest_mod.create_file(str(target), block, 2500, quiet=True, block_size=1000))
    read = iotest_mod.op_time(iotest_mod.read_file(str(target), None, 2500, quiet=True, block_size=1000))
    assert not isinstance(write, str), write
    assert not isinstance(read, str), read
    assert target.read_bytes() == block * 2 + block[:500]
    assert sum(consumed) == 2500


@pytest.mark.parametrize("op", ["write", "read"])
def test_a_stalled_chunk_names_the_bytes_moved_across_blocks(iotest_mod, tmp_path, monkeypatch, op):
    target = tmp_path / "f.bin"
    target.write_bytes(b"\x01" * 2500)
    calls = []

    def first_block_then_stall(fd, buffers):
        calls.append(1)
        if len(calls) > 1:
            return 0
        return (REAL_WRITEV if op == "write" else REAL_READV)(fd, buffers)

    monkeypatch.setattr(iotest_mod.os, op + "v", first_block_then_stall)
    if op == "write":
        result = iotest_mod.create_file(str(target), b"\x01" * 1000, 2500, quiet=True, block_size=1000)
    else:
        result = iotest_mod.read_file(str(target), None, 2500, quiet=True, block_size=1000)
    error = iotest_mod.op_time(result)
    assert isinstance(error, str)
    assert "after 1000 of 2500 bytes" in error


def test_chunked_stamping_is_inside_the_timed_write(iotest_mod, tmp_path, monkeypatch):
    clock = iter(range(100, 200))
    monkeypatch.setattr(iotest_mod.time, "perf_counter", lambda: next(clock))
    marks = []
    monkeypatch.setattr(iotest_mod, "stamp_chunk", lambda chunk, offset, **k: marks.append((offset, next(clock))))
    content = iotest_mod.FileContent(2500, True, True, 1, block_size=1000)
    start, end = content.write(str(tmp_path / "f.bin"), 0, quiet=True)
    assert [offset for offset, _ in marks] == [0, 1000, 2000]
    assert all(start < tick < end for _, tick in marks)


def test_whole_file_stamping_stays_before_the_timed_write(iotest_mod, tmp_path, monkeypatch):
    clock = iter(range(100, 200))
    monkeypatch.setattr(iotest_mod.time, "perf_counter", lambda: next(clock))
    marks = []
    monkeypatch.setattr(iotest_mod, "stamp_unique", lambda *a, **k: marks.append(next(clock)))
    content = iotest_mod.FileContent(2500, True, True, 1)
    start, end = content.write(str(tmp_path / "f.bin"), 0, quiet=True)
    assert marks == [100] and (start, end) == (101, 102)


@pytest.mark.parametrize("mode", ["write", "read", "random", "comprehensive"])
def test_workers_hold_block_sized_buffers(iotest_mod, work_dir, mode):
    if mode == "read":
        iotest_mod.worker(2, 4 * MiB, str(work_dir), [], "write", "0", True, True, 0)
    results = []
    # a whole-file buffer alone would be 4 MiB (comprehensive and random hold two)
    peak = traced_peak(iotest_mod.worker, 2, 4 * MiB, str(work_dir), results, mode, "0", True, False, 0,
        block_size=64 * KiB)
    assert len(results) == 2 and failures(results) == []
    assert peak < 1 * MiB


def test_simultaneous_and_layout_workers_hold_block_sized_buffers(iotest_mod, work_dir):
    peaks = [traced_peak(iotest_mod.layout_read_worker, 2, 4 * MiB, str(work_dir), "t", "0", True, False,
        block_size=64 * KiB)]
    for role in ("write", "read"):
        results = []
        peaks.append(traced_peak(iotest_mod.simultaneous_role_worker, role, 2, 4 * MiB, str(work_dir), "t", "0",
            results, True, False, 0, block_size=64 * KiB))
        assert len(results) == 2 and failures(results) == []
    assert all(peak < 1 * MiB for peak in peaks), peaks
    worker_dir = work_dir / ".iotest_simul_t" / "worker_0"
    assert sorted(p.stat().st_size for p in worker_dir.iterdir()) == [4 * MiB] * 4


@pytest.mark.parametrize("modes", [["write", "read"], ["random"], ["comprehensive"], ["simultaneous"]])
def test_every_mode_counts_whole_files_with_a_block_size(run_main, iotest_mod, modes):
    # an op is still one whole file, however many calls it takes
    result = run_main(modes, process_count=2, file_count=2, file_size=1024, block_size=300, unique=True)
    assert iotest_mod.total_failed_ops(result) == 0
    counts = {key: len(times) for key, times in result["outResults"].items()}
    if modes == ["random"]:
        assert sum(counts.values()) == 4
    elif modes == ["simultaneous"]:
        assert counts == {"write": 2, "read": 2}
    else:
        assert set(counts.values()) == {4}


def test_chunked_simultaneous_writer_and_layout_files_share_no_block(run_main, work_dir):
    run_main(["simultaneous"], process_count=2, file_count=2, file_size=8192, zeros=True, unique=True,
        block_size=1000, simultaneous_run_id="t", simultaneous_skip_cleanup=True)
    worker_dir = work_dir / ".iotest_simul_t" / "worker_0"
    files = [(worker_dir / name).read_bytes() for name in ("write_1.bin", "write_2.bin", "read_1.bin", "read_2.bin")]
    assert [len(data) for data in files] == [8192] * 4
    assert all_unique(*files)


def test_chunked_read_recreates_missing_files_whole_and_unique(run_main, iotest_mod, work_dir):
    result = run_main(["read"], file_count=2, file_size=8192, zeros=True, unique=True, block_size=1000)
    assert iotest_mod.total_failed_ops(result) == 0
    files = worker_files(work_dir)
    assert [len(data) for data in files] == [8192, 8192]
    assert all_unique(*files)


@pytest.mark.parametrize("block_size,expected_pc", [(1024, 4), (None, 1), (8192, 1), (16384, 1)])
def test_memory_estimate_uses_the_buffer_size(run_main, iotest_mod, monkeypatch, block_size, expected_pc):
    # 6000 B available: 4 workers x 1 KiB x 1.2 = 4915 B fit; one whole 8 KiB file per worker
    # (9830 B) does not, so without a smaller block the count drops to the minimum of 1
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (6000, 0))
    result = run_main(["write"], process_count=4, file_size=8192, block_size=block_size)
    assert result["process_count"] == expected_pc


class InlineProcess:
    # runs the target in this process, so a test can see what it was called with
    exitcode = 0

    def __init__(self, target, args):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)

    def join(self):
        pass


@pytest.mark.parametrize("block_size,size_text,gen_time", [
    (1024, "1.00 KiB blocks", "0.0010 s"), (None, "8.00 KiB files", "0.0080 s"), (16384, "8.00 KiB files", "0.0080 s")])
def test_generation_benchmark_uses_the_buffer_size(run_main, iotest_mod, rec_tl, monkeypatch, block_size, size_text, gen_time):
    # at 1024000 B/s, each worker's estimated generation time is its buffer size / 1024000
    def fake_benchmark(file_size, file_count, zeros, results, timeout=5):
        sizes.append(file_size)
        results.append(1024000)

    real_process = iotest_mod.Process

    def process(target, args):
        return InlineProcess(target, args) if target is fake_benchmark else real_process(target=target, args=args)

    sizes = []
    monkeypatch.setattr(iotest_mod, "benchmarkGenSpeed", fake_benchmark)
    monkeypatch.setattr(iotest_mod, "Process", process)
    run_main(["write"], file_size=8192, block_size=block_size, tl=rec_tl)
    assert sizes == [1024 if block_size == 1024 else 8192]
    assert any(size_text in line for line in rec_tl.printed)
    assert f"Gen time calculated:   \t{gen_time}" in rec_tl.printed


def test_block_size_at_least_the_file_size_is_unset(iotest_mod):
    assert iotest_mod.effective_block_size(None, 4096) is None
    assert iotest_mod.effective_block_size(4096, 4096) is None
    assert iotest_mod.effective_block_size(8192, 4096) is None
    assert iotest_mod.effective_block_size(1000, 4096) == 1000


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
@pytest.mark.parametrize("block_size", [4096, 8192])
def test_block_size_at_least_the_file_size_makes_the_same_calls_as_unset(run_main, iotest_mod, monkeypatch, tmp_path, block_size):
    def logged(kind, real):
        def call(fd, buffers):
            with open(log, "a") as f:
                f.write(f"{kind} {len(buffers[0])}\n")
            return real(fd, buffers)
        return call

    calls = {}
    for bs in (None, block_size):
        log = tmp_path / f"calls_{bs}.txt"
        # workers are forked, so they log to a file
        monkeypatch.setattr(iotest_mod.os, "writev", logged("write", REAL_WRITEV))
        monkeypatch.setattr(iotest_mod.os, "readv", logged("read", REAL_READV))
        run_main(["write", "read"], file_count=2, file_size=4096, unique=True, block_size=bs)
        calls[bs] = sorted(log.read_text().splitlines())
    assert calls[None] == calls[block_size] == ["read 4096"] * 2 + ["write 4096"] * 2


def test_tuning_runs_keep_the_block_size(iotest_mod, work_dir, rec_tl, tiny):
    # pc=1 with a 1 B/s target: one tuning attempt, accepted as best effort
    options = iotest_mod.RunOptions(file_size=8192, file_count=1, process_count=1, directories=str(work_dir),
        modes=["write"], zeros=False, no_report=True, benchmark_time=tiny["benchmarkTime"], block_size=1024)
    iotest_mod.tune_for_targets(options, rec_tl, 1, None)
    data = worker_files(work_dir, 1)[0]
    assert data == data[:1024] * 8


def test_simultaneous_tuning_lays_out_with_the_block_size(iotest_mod, work_dir, rec_tl, monkeypatch):
    class Stop(Exception):
        pass

    seen = []

    def spy(*args):
        # block_size, then verify
        seen.append(args[-2])
        raise Stop

    monkeypatch.setattr(iotest_mod, "run_simultaneous_layout", spy)
    options = iotest_mod.RunOptions(file_size=8192, file_count=1, process_count=2, directories=str(work_dir),
        modes=["simultaneous"], no_report=True, benchmark_time=1, block_size=1024)
    with pytest.raises(Stop):
        iotest_mod.tune_for_targets(options, rec_tl, 1, None)
    assert seen == [1024]


@pytest.mark.parametrize("args,header,json_value", [(["-bs", "1k"], "Block size: 1.00 KiB", 1024), ([], None, None)])
def test_cli_block_size(tmp_path, args, header, json_value):
    work = tmp_path / "w"
    out = tmp_path / "out.json"
    proc = run_iotest(["-fs", "8k", "-fc", "2", "-pc", "1", "-d", str(work), "-nl", "-nr", "-bt", "1", "-z", "-u",
                       "-j", str(out), *args, "write"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(out.read_text())["options"]["block_size"] == json_value
    if header:
        assert header in proc.stdout
    else:
        assert "Block size" not in proc.stdout
    assert all_unique(*worker_files(work))


@pytest.mark.parametrize("value", ["0", "0.4b", "abc"])
def test_cli_rejects_an_invalid_block_size(tmp_path, value):
    work = tmp_path / "w"
    proc = run_iotest(["-fs", "1k", "-fc", "1", "-pc", "1", "-bs", value, "-d", str(work), "-nl", "-nr", "-bt", "1", "write"])
    # argparse's usage error, before anything runs
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "-bs" in proc.stderr and "Traceback" not in proc.stderr
    assert not work.exists()
