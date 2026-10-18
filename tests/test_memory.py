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
    "modes,read_recreates,copies",
    [
        (["write"], False, 1),
        (["index"], False, 1),
        (["benchmark"], False, 1),
        # read workers hold only their read buffer, unless they have files to recreate
        (["read"], False, 1),
        (["read"], True, 2),
        (["write", "read"], False, 1),
        (["simultaneous"], False, 1),
        # these workers write and read, so they hold file content and a read buffer
        (["comprehensive"], False, 2),
        (["random"], False, 2),
    ],
)
def test_memory_copies_per_mode(iotest_mod, modes, read_recreates, copies):
    assert iotest_mod.memory_copies(modes, read_recreates) == copies


@pytest.mark.parametrize(
    "modes,prepared,expected",
    [
        (["read"], True, False),            # every file is there at full size
        (["read"], False, True),            # fresh directory: read recreates them
        (["write", "read"], False, False),  # the write pass creates them first
        (["read", "write"], False, True),   # ...but not when it runs after the read
        (["write"], False, False),
    ],
)
def test_read_recreates_files(iotest_mod, work_dir, run_main, modes, prepared, expected):
    if prepared:
        run_main(["write"], process_count=2, file_count=2)
    assert iotest_mod.read_recreates_files(modes, [str(work_dir)], 2, 2, 1024) == expected


@pytest.mark.parametrize("prepared", [True, False])
def test_memory_guard_counts_read_buffers(run_main, iotest_mod, monkeypatch, prepared):
    # just enough for 4 read workers that hold only a read buffer (every file is present); on a
    # fresh directory they also need content to recreate the files, so fewer fit
    read_only = iotest_mod.memory_per_process(["read"], 1024, 1)
    recreating = iotest_mod.memory_per_process(["read"], 1024, 1, read_recreates=True)
    available = int(4 * read_only) + 1
    expected_pc = 4 if prepared else int(available // recreating)
    assert expected_pc < 4 or prepared
    if prepared:
        run_main(["write"], process_count=4)
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (available, 0))
    result = run_main(["read"], process_count=4)
    assert result["process_count"] == expected_pc


def test_memory_cut_keeps_at_least_one_simultaneous_reader(run_main, iotest_mod, monkeypatch):
    # memory for 2.5 simultaneous workers: 2 processes fit, fewer than the 3 writers asked for;
    # the split must shrink to 1 + 1, not fail with 0 readers
    available = int(2.5 * iotest_mod.memory_per_process(["simultaneous"], 1024, 1))
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (available, 0))
    result = run_main(["simultaneous"], process_count=4, simultaneous_write_count=3)
    assert result["process_count"] == 2
    assert result["mode_process_counts"] == {"write": 1, "read": 1}


def test_read_worker_holds_only_its_read_buffer(iotest_mod, work_dir):
    iotest_mod.worker(2, 4 * MiB, str(work_dir), [], "write", "0", True, True, 0)
    # every file exists, so no random content is generated for recreating them
    peak = traced_peak(iotest_mod.worker, 2, 4 * MiB, str(work_dir), [], "read", "0", True, False, 0)
    assert peak < 1.5 * 4 * MiB


def test_read_worker_recreates_missing_files_on_demand(iotest_mod, work_dir):
    results = []
    iotest_mod.worker(2, 4096, str(work_dir), results, "read", "0", True, False, 0)
    assert len(results) == 2 and not any(isinstance(r, str) for r in results)
    assert [p.stat().st_size for p in sorted((work_dir / "0").glob("temp_*.bin"))] == [4096, 4096]


def test_index_worker_holds_no_file_data(iotest_mod, work_dir):
    peak = traced_peak(iotest_mod.worker, 2, 4 * MiB, str(work_dir), [], "index", "0", True, False, 0)
    assert peak < 0.5 * 4 * MiB


def test_simultaneous_reader_holds_only_its_read_buffer(iotest_mod, work_dir, rec_tl):
    iotest_mod.run_simultaneous_layout(2, 4 * MiB, [str(work_dir)], "t", 1, 1, True, True, rec_tl)
    peak = traced_peak(iotest_mod.simultaneous_role_worker,
        "read", 2, 4 * MiB, str(work_dir), "t", "0", [], True, False, 0)
    assert peak < 1.5 * 4 * MiB


@pytest.mark.parametrize(
    "modes,prepared,skipped",
    [
        (["read"], True, True),
        # a fresh directory: read workers generate content to recreate files up front
        (["read"], False, False),
        (["index"], False, True),
        (["write"], False, False),
        (["write", "read"], False, False),
    ],
)
def test_generation_benchmark_runs_only_when_workers_generate_data(run_main, rec_tl, modes, prepared, skipped):
    # it estimates how long workers take to generate their data before the start barrier
    if prepared:
        run_main(["write"])
    run_main(modes, tl=rec_tl)
    assert any("Skipping file generation benchmark" in m for m in rec_tl.printed) == skipped


def test_read_worker_that_will_recreate_generates_before_the_start_barrier(iotest_mod, work_dir, monkeypatch):
    # generating inside the timed window would count CPU work as read time
    made_at = []
    real = iotest_mod.make_file_content
    monkeypatch.setattr(iotest_mod, "make_file_content",
        lambda *a, **k: made_at.append(iotest_mod.time.perf_counter()) or real(*a, **k))
    start = iotest_mod.time.perf_counter() + 0.3
    iotest_mod.worker(1, 4096, str(work_dir), [], "read", "0", True, True, start, False, True)
    assert len(made_at) == 1
    assert made_at[0] < start


def test_memory_cut_that_changes_padding_recounts_recreated_files(run_main, iotest_mod, monkeypatch):
    # a -pc 12 write leaves 00..11. The memory fits 9 read-only workers, but at -pc 9 the padding
    # is 0..8, those files are missing, and each worker would also hold content to recreate them
    read_only = iotest_mod.memory_per_process(["read"], 1024, 1)
    recreating = iotest_mod.memory_per_process(["read"], 1024, 1, read_recreates=True)
    available = int(9 * read_only) + 1
    assert int(available // recreating) < 9
    run_main(["write"], process_count=12)
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (available, 0))
    result = run_main(["read"], process_count=12)
    assert result["process_count"] == int(available // recreating)


@pytest.mark.parametrize(
    "modes,ops_per_file",
    [
        (["write"], 1),
        (["read"], 1),
        (["random"], 1),
        (["index"], 1),
        (["comprehensive"], 4),
        # three passes of one op
        (["meta"], 3),
        # each simultaneous worker, writer or reader, does one op per file
        (["simultaneous"], 1),
        (["benchmark"], 0),
        (["write", "read", "index"], 3),
    ],
)
def test_ops_per_process_sums_the_passes(iotest_mod, modes, ops_per_file):
    assert iotest_mod.ops_per_process(modes, 7) == 7 * ops_per_file


def test_memory_per_process_grows_with_file_count_and_timeline(iotest_mod):
    per_process = iotest_mod.memory_per_process
    result, timeline = iotest_mod.RESULT_BYTES_PER_OP, iotest_mod.TIMELINE_BYTES_PER_OP
    assert result > 0 and timeline > 0
    # the data buffers, as before
    buffers = per_process(["write"], 1024, 0)
    assert buffers == pytest.approx(1024 * 1.2)
    assert per_process(["comprehensive"], 1024, 0) == pytest.approx(2 * 1024 * 1.2)
    # plus every op the process performs over the run
    assert per_process(["write"], 1024, 1000) == pytest.approx(buffers + 1000 * result)
    assert per_process(["write", "read"], 1024, 1000) == pytest.approx(buffers + 2000 * result)
    assert per_process(["write"], 1024, 1000, timeline=True) == pytest.approx(buffers + 1000 * (result + timeline))
    assert per_process(["comprehensive"], 1024, 1000, timeline=True) == pytest.approx(
        2 * 1024 * 1.2 + 4000 * (result + timeline))
    # meta holds no data, but no longer estimates zero
    assert per_process(["meta"], 1024, 1000) == pytest.approx(3000 * result)
    assert per_process(["meta"], 1024, 1000, timeline=True) == pytest.approx(3000 * (result + timeline))


@pytest.mark.parametrize("timeline", [False, True])
def test_memory_guard_counts_the_ops_of_a_meta_run(run_main, iotest_mod, monkeypatch, rec_tl, timeline):
    # room for 2.5 processes' results (and timeline): 2 of the 4 run
    per_process = iotest_mod.memory_per_process(["meta"], 1024, 100, timeline=timeline)
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (int(2.5 * per_process), 0))
    result = run_main(["meta"], process_count=4, file_count=100, tl=rec_tl, timeline=timeline)
    assert result["process_count"] == 2
    assert f"Estimated total memory usage: {iotest_mod.format_bytes(4 * per_process)}B" in rec_tl.printed
    assert "Reducing the number of processes to 2" in rec_tl.printed
