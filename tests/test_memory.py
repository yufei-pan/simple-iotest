import os
import tracemalloc

import pytest

MiB = 2**20


def traced_peak(fn, *args):
    tracemalloc.start()
    try:
        fn(*args)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def generators(iotest_mod):
    yield False
    if iotest_mod.numpy_available:
        yield True


def test_almost_urandom_holds_one_copy(iotest_mod, monkeypatch):
    monkeypatch.setattr(iotest_mod, "GEN_CHUNK", 1 * MiB, raising=False)
    for use_numpy in generators(iotest_mod):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        # a second full-size copy (numpy array + .tobytes(), or int + to_bytes) would be 2x
        assert traced_peak(iotest_mod.almost_urandom, 8 * MiB) < 1.5 * 8 * MiB


def test_almost_urandom_chunks_continue_the_stream(iotest_mod, monkeypatch):
    monkeypatch.setattr(iotest_mod, "GEN_CHUNK", 1024, raising=False)
    for use_numpy in generators(iotest_mod):
        monkeypatch.setattr(iotest_mod, "numpy_available", use_numpy)
        buf = iotest_mod.almost_urandom(4 * 1024 + 100)
        assert isinstance(buf, bytearray)
        assert len(buf) == 4 * 1024 + 100
        chunks = [bytes(buf[o:o + 1024]) for o in range(0, 4 * 1024, 1024)]
        assert len(set(chunks)) == 4


def test_generation_benchmark_holds_one_copy(iotest_mod, monkeypatch):
    monkeypatch.setattr(iotest_mod, "GEN_CHUNK", 1 * MiB, raising=False)
    results = []
    # 3 generations; keeping the previous buffer alive while making the next is 2-3x
    peak = traced_peak(iotest_mod.benchmarkGenSpeed, 8 * MiB, 3, False, results, 60)
    assert results
    assert peak < 1.5 * 8 * MiB


def test_read_file_reads_into_the_given_buffer(iotest_mod, tmp_path):
    content = os.urandom(2500)
    target = tmp_path / "f.bin"
    target.write_bytes(content)
    buffer = bytearray(2500)
    op = iotest_mod.op_time(iotest_mod.read_file(str(target), content, 2500, quiet=True, buffer=buffer))
    assert not isinstance(op, str), op
    assert bytes(buffer) == content


def spy_read_buffers(iotest_mod, monkeypatch):
    seen = []
    real = iotest_mod.read_file

    def spy(*args, **kwargs):
        seen.append(kwargs.get("buffer"))
        return real(*args, **kwargs)

    monkeypatch.setattr(iotest_mod, "read_file", spy)
    return seen


@pytest.mark.parametrize("mode", ["read", "comprehensive"])
def test_worker_reuses_one_read_buffer(iotest_mod, work_dir, monkeypatch, mode):
    seen = spy_read_buffers(iotest_mod, monkeypatch)
    iotest_mod.worker(3, 4096, str(work_dir), [], mode, "0", True, True, 0)
    assert len(seen) == 3
    assert seen[0] is not None
    assert all(b is seen[0] for b in seen)


def test_simultaneous_reader_reuses_one_read_buffer(iotest_mod, work_dir, monkeypatch):
    seen = spy_read_buffers(iotest_mod, monkeypatch)
    iotest_mod.simultaneous_role_worker("read", 3, 4096, str(work_dir), "t", "0", [], True, True, 0)
    assert len(seen) == 3
    assert seen[0] is not None
    assert all(b is seen[0] for b in seen)


@pytest.mark.parametrize(
    "modes,copies",
    [
        (["write"], 1),
        (["index"], 1),
        (["benchmark"], 1),
        (["read"], 2),
        (["write", "read"], 2),
        (["comprehensive"], 2),
        (["random"], 2),
        (["simultaneous"], 2),
    ],
)
def test_memory_copies_per_mode(iotest_mod, modes, copies):
    # every worker holds its file content; workers that also read hold a read buffer too
    assert iotest_mod.memory_copies(modes) == copies


@pytest.mark.parametrize("modes,expected_pc", [(["read"], 2), (["write"], 4)])
def test_memory_guard_counts_read_buffers(run_main, iotest_mod, monkeypatch, modes, expected_pc):
    # 6000 B available: 4 workers x 1024 B x 1.2 = 4915 B fits for write-only,
    # but read workers hold two copies (9830 B), so only 6000 // (1024 x 2 x 1.2) = 2 fit
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (6000, 0), raising=False)
    if modes == ["read"]:
        run_main(["write"], process_count=4)
    result = run_main(modes, process_count=4)
    assert result["process_count"] == expected_pc


def test_memory_cut_keeps_at_least_one_simultaneous_reader(run_main, iotest_mod, monkeypatch):
    # 7400 B fits 7400 // (1024 x 2 x 1.2) = 3 processes, but 3 writers were asked for;
    # the split must shrink to 2 writers + 1 reader instead of failing with 0 readers
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (7400, 0))
    result = run_main(["simultaneous"], process_count=4, simultaneous_write_count=3)
    assert result["process_count"] == 3
    assert result["mode_process_counts"] == {"write": 2, "read": 1}
