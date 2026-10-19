import ctypes
import functools
import json
import os
import tracemalloc
import zlib

import pytest

from test_cli import run_iotest

KiB = 2**10
MiB = 2**20
# Windows has neither; the tests that use them skip there
REAL_WRITEV = getattr(os, "writev", None)
REAL_READV = getattr(os, "readv", None)


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
    return [(work_dir / "0000" / f"temp_{i}.bin").read_bytes() for i in range(1, count + 1)]


def worker_samples(iotest_mod, results):
    # a worker hands over one compact record per pass; comprehensive's holds a write, move, stat and read per file
    (record,) = results
    return list(iotest_mod.result_samples([record]))


def traced_peak(fn, *args, **kwargs):
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("zeros", [True, False])
@pytest.mark.parametrize("block_size", [1024, 1000])
# a file the chunk pool holds, and one larger than the pool, whose chunks come from moving starts in it
@pytest.mark.parametrize("file_size", [8192 + 100, 2 * MiB + 100])
def test_chunked_unique_files_keep_their_size_and_never_repeat_a_block(iotest_mod, tmp_path, monkeypatch, zeros, block_size, file_size):
    # 1000 is not a multiple of 512, so stamps straddle the chunks
    for use_numpy in stamp_paths(iotest_mod):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        d = tmp_path / str(use_numpy)
        d.mkdir()
        iotest_mod.worker(2, file_size, str(d), [], "write", "0000", True, zeros, 0, True, block_size=block_size)
        files = worker_files(d)
        assert [len(data) for data in files] == [file_size, file_size]
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


@pytest.mark.parametrize("blocks", [4, 200])
def test_stamp_unique_numbers_blocks_from_first_block(iotest_mod, monkeypatch, blocks):
    # 200 blocks: numpy stamps them when it is there
    for use_numpy in stamp_paths(iotest_mod):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        whole = bytearray(blocks * 512 + 100)
        iotest_mod.stamp_unique(whole, 9, 3)
        part = bytearray((blocks - 2) * 512 + 100)
        iotest_mod.stamp_unique(memoryview(part), 9, 3, first_block=2)
        assert bytes(part) == bytes(whole[2 * 512:])


@pytest.mark.parametrize("tail", [0, 100])
@pytest.mark.parametrize("more", [-1, 0, 1])
def test_numpy_and_struct_stamping_give_the_same_bytes_at_the_threshold(iotest_mod, monkeypatch, more, tail):
    pytest.importorskip("numpy")
    size = (iotest_mod.STAMP_NUMPY_MIN_BLOCKS + more) * 512 + tail
    base = os.urandom(size)
    stamped = []
    for use_numpy in (False, True):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        buf = bytearray(base)
        iotest_mod.stamp_unique(memoryview(buf)[:size], 0x0102030405060708, 77, first_block=5)
        stamped.append(bytes(buf))
    assert stamped[0] == stamped[1]
    assert stamped[0] != base


@pytest.mark.parametrize("more,uses_numpy", [(-1, False), (0, True)])
def test_chunks_of_fewer_blocks_than_the_threshold_are_stamped_without_numpy(iotest_mod, monkeypatch, more, uses_numpy):
    # numpy costs more than it saves on small chunks, stamped inside the timed write
    class NoNumpy:
        def __getattr__(self, name):
            raise AssertionError("numpy used")

    monkeypatch.setattr(iotest_mod, "numpy_available", True)
    monkeypatch.setattr(iotest_mod, "np", NoNumpy(), raising=False)
    buf = bytearray((iotest_mod.STAMP_NUMPY_MIN_BLOCKS + more) * 512)
    if uses_numpy:
        with pytest.raises(AssertionError, match="numpy used"):
            iotest_mod.stamp_unique(buf, 1, 2)
    else:
        iotest_mod.stamp_unique(buf, 1, 2)
        assert buf[:24] == iotest_mod.UNIQUE_TAG.pack(2, 0, 1)


def compression_ratio(data):
    return len(data) / len(zlib.compress(data, 1))


@pytest.mark.parametrize("unique", [False, True])
def test_a_chunked_file_does_not_compress(run_main, work_dir, unique):
    # chunks come from a pool of random data, not one block written again and again, which compressing storage
    # would store in a fraction of the bytes; a file the pool holds is the pool itself
    run_main(["write"], file_size=256 * KiB, zeros=False, unique=unique, block_size=4 * KiB)
    data = worker_files(work_dir, 1)[0]
    assert len(data) == 256 * KiB
    assert compression_ratio(data) < 1.1


LZMA_FILTERS = [{"id": 33, "preset": 0, "dict_size": 4 * MiB}]  # 33: lzma.FILTER_LZMA2


def lzma_size(lzma, data):
    return len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS))


@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("block_size", [4 * KiB, 1000, 128 * KiB, 1 * MiB])
def test_no_mib_of_a_file_larger_than_the_pool_compresses(iotest_mod, tmp_path, unique, block_size):
    # storage compresses records of up to about 1 MiB (ZFS recordsize, btrfs 128 KiB extents, arrays), so no
    # chunk may repeat the bytes of another within any 1 MiB of the file, aligned or not. LZMA with a dictionary
    # larger than the window finds any repeat in it. -u's tags compress a little whatever the block size, so the
    # file written whole is the reference
    lzma = pytest.importorskip("lzma")
    file_size = 4 * MiB
    assert iotest_mod.content_size(file_size, block_size) < file_size
    data = {}
    for bs in (None, block_size):
        d = tmp_path / str(bs)
        iotest_mod.worker(1, file_size, str(d), [], "write", "0000", True, False, 0, unique, block_size=bs)
        data[bs] = (d / "0000" / "temp_1.bin").read_bytes()
        assert len(data[bs]) == file_size
    for start in (0, 1 * MiB, 1536 * KiB + 777, 3 * MiB):
        whole, chunked = (lzma_size(lzma, data[bs][start:start + MiB]) for bs in (None, block_size))
        assert chunked > 0.995 * whole, start


def test_lzma_finds_a_chunk_repeated_a_mib_apart():
    # the check above would see a chunk that repeats an earlier one, even shifted by a few bytes
    lzma = pytest.importorskip("lzma")
    pool = os.urandom(MiB)
    window = pool[:7 * 128 * KiB] + pool[3:128 * KiB + 3]
    assert lzma_size(lzma, window) < 0.9 * lzma_size(lzma, pool)


@pytest.mark.parametrize("file_size,block_size,pool", [
    # a file the pool would hold is held whole, so it is the pool itself
    (256 * KiB, 4 * KiB, 256 * KiB),
    # 1 MiB of chunk starts and one block
    (8 * MiB, 4 * KiB, MiB + 4 * KiB),
    (8 * MiB, 3 * MiB, 4 * MiB),
    (8 * MiB, None, 8 * MiB),
])
def test_content_size(iotest_mod, file_size, block_size, pool):
    assert iotest_mod.content_size(file_size, block_size) == pool


def pool_starts(pool, chunks):
    # where in the pool each chunk's view starts
    base = ctypes.addressof(ctypes.c_char.from_buffer(pool))
    return [ctypes.addressof(ctypes.c_char.from_buffer(chunk)) - base for _, chunk in chunks]


@pytest.mark.parametrize("stamped", [False, True])
@pytest.mark.parametrize("block_size", [7, 512, 1000, 4 * KiB, 128 * KiB, MiB, 3 * MiB])
def test_a_pool_byte_comes_back_in_the_file_only_a_mib_or_more_later(iotest_mod, block_size, stamped):
    file_size = 3 * iotest_mod.content_size(8 * MiB, block_size) + 100
    pool = bytearray(iotest_mod.content_size(file_size, block_size))
    chunks = list(iotest_mod.file_chunks(pool, file_size, block_size, stamped=stamped))
    assert [offset for offset, _ in chunks] == list(range(0, file_size, block_size))
    assert [len(chunk) for _, chunk in chunks] == [min(block_size, file_size - offset) for offset, _ in chunks]
    # each chunk starts at its offset in the file modulo the span: file bytes that share a pool byte are a
    # multiple of the span apart, and no chunk runs off the pool
    span = iotest_mod.chunk_span(len(pool), block_size, file_size, stamped)
    assert pool_starts(pool, chunks) == [offset % span for offset, _ in chunks]
    assert iotest_mod.CHUNK_POOL_SIZE <= span <= len(pool) - block_size + 1
    if stamped:
        # -u tags land on the same bytes of the pool whichever chunk stamps them
        assert span % 512 == 0
    else:
        # each round of the pool sits at another place in the file's 512-byte blocks
        assert span % 2 == 1


def test_a_one_block_buffer_gives_every_chunk_from_its_start(iotest_mod):
    for stamped in (False, True):
        block = bytearray(1000)
        chunks = list(iotest_mod.file_chunks(block, 2500, 1000, stamped=stamped))
        assert [offset for offset, _ in chunks] == [0, 1000, 2000]
        assert pool_starts(block, chunks) == [0, 0, 0]
        assert [len(chunk) for _, chunk in chunks] == [1000, 1000, 500]


def test_a_buffer_that_holds_the_file_gives_each_chunk_from_its_own_place(iotest_mod):
    pool = bytearray(2500)
    chunks = list(iotest_mod.file_chunks(pool, 2500, 1000))
    assert pool_starts(pool, chunks) == [0, 1000, 2000]
    assert [len(chunk) for _, chunk in chunks] == [1000, 1000, 500]


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
    op = iotest_mod.op_time(iotest_mod.read_file(str(target), 2500, quiet=True, buffer=buffer, block_size=1000))
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
    iotest_mod.read_file(str(target), 2500, quiet=True, block_size=1000)
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
    read = iotest_mod.op_time(iotest_mod.read_file(str(target), 2500, quiet=True, block_size=1000))
    assert not isinstance(write, str), write
    assert not isinstance(read, str), read
    assert target.read_bytes() == block * 2 + block[:500]
    assert sum(consumed) == 2500


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
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
        result = iotest_mod.read_file(str(target), 2500, quiet=True, block_size=1000)
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
def test_workers_hold_a_chunk_pool_and_block_sized_read_buffers(iotest_mod, work_dir, mode):
    if mode == "read":
        iotest_mod.worker(2, 8 * MiB, str(work_dir), [], "write", "0", True, True, 0)
    results = []
    # a whole-file buffer alone would be 8 MiB (comprehensive and random hold two); the chunk pool is 1 MiB and one
    # block, and making it briefly takes up to twice as much again
    peak = traced_peak(iotest_mod.worker, 2, 8 * MiB, str(work_dir), results, mode, "0", True, False, 0,
        block_size=64 * KiB)
    samples = worker_samples(iotest_mod, results)
    assert len(samples) == (8 if mode == "comprehensive" else 2)
    assert not [s for s in samples if isinstance(s, str)]
    assert peak < (1 if mode == "read" else 4) * MiB


def test_simultaneous_and_layout_workers_hold_a_chunk_pool_or_a_block_sized_read_buffer(iotest_mod, work_dir):
    peaks = [traced_peak(iotest_mod.layout_read_worker, 2, 8 * MiB, str(work_dir), "t", "0", True, False,
        block_size=64 * KiB)]
    for role in ("write", "read"):
        results = []
        peaks.append(traced_peak(iotest_mod.simultaneous_role_worker, role, 2, 8 * MiB, str(work_dir), "t", "0",
            results, True, False, 0, block_size=64 * KiB))
        samples = worker_samples(iotest_mod, results)
        assert len(samples) == 2 and not [s for s in samples if isinstance(s, str)]
    # the layout worker and the writer hold a chunk pool (see above), the reader a block
    assert all(peak < 4 * MiB for peak in peaks), peaks
    assert peaks[2] < 1 * MiB
    worker_dir = work_dir / ".iotest_simul_t" / "worker_0"
    assert sorted(p.stat().st_size for p in worker_dir.iterdir()) == [8 * MiB] * 4


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
    worker_dir = work_dir / ".iotest_simul_t" / "worker_0000"
    files = [(worker_dir / name).read_bytes() for name in ("write_1.bin", "write_2.bin", "read_1.bin", "read_2.bin")]
    assert [len(data) for data in files] == [8192] * 4
    assert all_unique(*files)


def test_chunked_read_recreates_missing_files_whole_and_unique(run_main, iotest_mod, work_dir):
    result = run_main(["read"], file_count=2, file_size=8192, zeros=True, unique=True, block_size=1000)
    assert iotest_mod.total_failed_ops(result) == 0
    files = worker_files(work_dir)
    assert [len(data) for data in files] == [8192, 8192]
    assert all_unique(*files)


@pytest.mark.parametrize("block_size,expected_pc", [(1024, 4), (None, 1), (8 * MiB, 1), (16 * MiB, 1)])
def test_memory_estimate_uses_the_buffer_size(run_main, iotest_mod, monkeypatch, block_size, expected_pc):
    # 6 MiB available: 4 workers x (1 MiB + 2 KiB chunk pool) x 1.2 = 4.8 MiB fit; one whole 8 MiB file per worker
    # (9.6 MiB) does not, so without a smaller block the count drops to the minimum of 1
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (6 * MiB, 0))
    result = run_main(["write"], process_count=4, file_size=8 * MiB, block_size=block_size)
    assert result["process_count"] == expected_pc


def test_memory_estimate_counts_the_chunk_pool_and_the_block_sized_read_buffer(iotest_mod):
    pool = iotest_mod.content_size(64 * MiB, 4 * KiB)
    per_process = functools.partial(iotest_mod.memory_per_process, file_count=0, read_size=4 * KiB)
    assert per_process(["write"], pool) == pytest.approx(pool * 1.2)
    assert per_process(["read"], pool) == pytest.approx(4 * KiB * 1.2)
    assert per_process(["random"], pool) == pytest.approx((pool + 4 * KiB) * 1.2)
    assert per_process(["read"], pool, read_recreates=True) == pytest.approx((pool + 4 * KiB) * 1.2)


class InlineProcess:
    # runs the target in this process, so a test can see what it was called with
    exitcode = 0

    def __init__(self, target, args):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)

    def join(self):
        pass


@pytest.mark.parametrize("file_size,block_size,size,size_text", [
    (8192, 1024, 8192, "8.00 KiB chunk pools"), (4 * MiB, 1024, MiB + 1024, "1.00 MiB chunk pools"),
    (8192, None, 8192, "8.00 KiB files"), (8192, 16384, 8192, "8.00 KiB files")])
def test_generation_benchmark_uses_the_buffer_size(run_main, iotest_mod, rec_tl, monkeypatch, file_size, block_size, size, size_text):
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
    run_main(["write"], file_size=file_size, block_size=block_size, tl=rec_tl)
    assert sizes == [size]
    assert any(size_text in line for line in rec_tl.printed)
    assert f"Gen time calculated:   \t{size / 1024000:.4f} s" in rec_tl.printed


def test_block_size_at_least_the_file_size_is_unset(iotest_mod):
    assert iotest_mod.effective_block_size(None, 4096) is None
    assert iotest_mod.effective_block_size(4096, 4096) is None
    assert iotest_mod.effective_block_size(8192, 4096) is None
    assert iotest_mod.effective_block_size(1000, 4096) == 1000


@pytest.mark.patches_workers
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


@pytest.mark.patches_workers
@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_tuning_runs_keep_the_block_size(iotest_mod, work_dir, rec_tl, tiny, monkeypatch, tmp_path):
    log = tmp_path / "writes.txt"

    def logged(fd, buffers):
        # workers are forked, so they log to a file
        with open(str(log), "a") as f:
            f.write(f"{len(buffers[0])}\n")
        return REAL_WRITEV(fd, buffers)

    monkeypatch.setattr(iotest_mod.os, "writev", logged)
    # pc=1 with a 1 B/s target: one tuning attempt, accepted as best effort
    options = iotest_mod.RunOptions(file_size=8192, file_count=1, process_count=1, directories=str(work_dir),
        modes=["write"], zeros=False, no_report=True, benchmark_time=tiny["benchmarkTime"], block_size=1024)
    iotest_mod.tune_for_targets(options, rec_tl, 1, None)
    assert log.read_text().split() == ["1024"] * 8
    assert len(worker_files(work_dir, 1)[0]) == 8192


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
