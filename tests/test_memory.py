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
    "modes,read_recreates,contents,reads",
    [
        (["write"], False, 1, 0),
        # index and meta move no data
        (["index"], False, 0, 0),
        (["meta"], False, 0, 0),
        (["index", "meta"], False, 0, 0),
        (["index", "write"], False, 1, 0),
        (["meta", "read"], True, 1, 1),
        # the generation benchmark's processes each hold file content
        (["benchmark"], False, 1, 0),
        # read workers hold only their read buffer, unless they have files to recreate
        (["read"], False, 0, 1),
        (["read"], True, 1, 1),
        # the write pass holds more: file content is never smaller than a read buffer
        (["write", "read"], False, 1, 0),
        # a writer or a reader, counted as a writer
        (["simultaneous"], False, 1, 0),
        # these workers write and read, so they hold file content and a read buffer
        (["comprehensive"], False, 1, 1),
        (["random"], False, 1, 1),
    ],
)
def test_buffer_memory_per_mode(iotest_mod, modes, read_recreates, contents, reads):
    # the most any one pass holds: file content (the whole file, or with -bs the chunk pool) and read buffers
    # (the whole file, or one block) count at their own sizes
    assert iotest_mod.buffer_memory(modes, 1000, 1, read_recreates) == contents * 1000 + reads
    assert iotest_mod.buffer_memory(modes, 1, 1, read_recreates) == contents + reads


@pytest.mark.parametrize(
    "modes,expected",
    [
        # as found: worker 1 has a missing or wrong-size file
        (["read"], {0: [False, True, False]}),
        # the write pass creates the files first...
        (["write", "read"], {1: [False] * 3}),
        # ...but not when it runs after the read
        (["read", "write"], {0: [False, True, False]}),
        # a comprehensive pass renames and deletes every file, and the last pass before a read decides
        (["comprehensive", "read"], {1: [True] * 3}),
        (["write", "comprehensive", "read"], {2: [True] * 3}),
        (["comprehensive", "write", "read"], {2: [False] * 3}),
        (["random", "read"], {1: [False] * 3}),
        # a read pass recreates what it does not find, so a later one finds everything
        (["read", "read"], {0: [False, True, False], 1: [False] * 3}),
        (["read", "comprehensive", "read"], {0: [False, True, False], 2: [True] * 3}),
        (["index", "meta", "read"], {2: [False, True, False]}),
        (["write"], {}),
    ],
)
def test_read_pass_recreates(iotest_mod, modes, expected):
    scan = [(1024, 0), (0, 1), (1024, 0), (0, 1)]
    assert iotest_mod.read_pass_recreates(modes, scan, 3) == expected


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
    samples = list(iotest_mod.result_samples(results))
    assert len(samples) == 2 and not any(isinstance(s, str) for s in samples)
    assert [p.stat().st_size for p in sorted((work_dir / "0").glob("temp_*.bin"))] == [4096, 4096]


def test_index_worker_holds_no_file_data(iotest_mod, work_dir):
    peak = traced_peak(iotest_mod.worker, 2, 4 * MiB, str(work_dir), [], "index", "0", True, False, 0)
    assert peak < 0.5 * 4 * MiB


def test_simultaneous_reader_holds_only_its_read_buffer(iotest_mod, work_dir, rec_tl):
    iotest_mod.run_simultaneous_layout(2, 4 * MiB, [str(work_dir)], "t", 1, True, True, rec_tl)
    peak = traced_peak(iotest_mod.simultaneous_role_worker,
        "read", 2, 4 * MiB, str(work_dir), "t", iotest_mod.worker_counter_name(0), [], True, False, 0)
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


def test_memory_cut_keeps_the_files_of_the_first_workers(run_main, iotest_mod, monkeypatch):
    # a -pc 12 write leaves 0000..0011. The memory fits 9 read-only workers: they are 0000..0008,
    # whose files are all there, so none of them holds content to recreate files
    read_only = iotest_mod.memory_per_process(["read"], 1024, 1)
    available = int(9 * read_only) + 1
    run_main(["write"], process_count=12)
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (available, 0))
    result = run_main(["read"], process_count=12)
    assert result["process_count"] == 9


def test_worker_names_do_not_depend_on_the_process_count(iotest_mod):
    assert [iotest_mod.worker_counter_name(c) for c in (0, 9, 11, 9999, 12345)] == [
        "0000", "0009", "0011", "9999", "12345"]
    assert [name for _, name in iotest_mod.worker_dirs(["d"], 3)] == ["0000", "0001", "0002"]


def test_a_read_at_a_smaller_process_count_finds_the_files_of_a_larger_write(run_main, work_dir, rec_tl):
    # -t 12 then -t 9 used to look for 0..8 instead of the 00..11 the write left, and recreate every
    # file inside the timed read pass
    run_main(["write"], process_count=12, file_count=2)
    written = {p: p.stat().st_mtime_ns for p in work_dir.rglob("temp_*.bin")}
    assert sorted({p.parent.name for p in written}) == [f"{c:04d}" for c in range(12)]
    result = run_main(["read"], process_count=9, file_count=2, tl=rec_tl)
    assert len(result["outResults"]["read"]) == 18
    assert {p: p.stat().st_mtime_ns for p in work_dir.rglob("temp_*.bin")} == written
    # nothing to recreate, so no content to generate
    assert any("Skipping file generation benchmark" in m for m in rec_tl.printed)


def test_every_mode_has_its_per_op_figures(iotest_mod):
    modes = set(iotest_mod.OPS_PER_FILE)
    assert modes == {"write", "read", "random", "index", "comprehensive", "meta", "simultaneous"}
    assert set(iotest_mod.RESULT_BYTES_PER_OP) == set(iotest_mod.TIMELINE_BYTES_PER_OP) == modes
    assert all(v > 0 for v in iotest_mod.RESULT_BYTES_PER_OP.values())
    assert all(v > 0 for v in iotest_mod.TIMELINE_BYTES_PER_OP.values())


@pytest.mark.parametrize(
    "mode,ops_per_file",
    [
        ("write", 1),
        ("read", 1),
        ("random", 1),
        ("index", 1),
        ("comprehensive", 4),
        # three passes of one op
        ("meta", 3),
        # each simultaneous worker, writer or reader, does one op per file
        ("simultaneous", 1),
    ],
)
def test_memory_per_process_counts_every_op_at_its_modes_figure(iotest_mod, mode, ops_per_file):
    per_process = iotest_mod.memory_per_process
    result, timeline = iotest_mod.RESULT_BYTES_PER_OP[mode], iotest_mod.TIMELINE_BYTES_PER_OP[mode]
    buffers = per_process([mode], 1024, 0)
    # grows with the file count, and more with --timeline
    assert per_process([mode], 1024, 1000) == pytest.approx(buffers + 1000 * ops_per_file * result)
    assert per_process([mode], 1024, 1000, timeline=True) == pytest.approx(
        buffers + 1000 * ops_per_file * (result + timeline))


def test_memory_per_process_adds_the_modes_ops_to_the_buffers(iotest_mod):
    per_process = iotest_mod.memory_per_process
    result, timeline = iotest_mod.RESULT_BYTES_PER_OP, iotest_mod.TIMELINE_BYTES_PER_OP
    # the data buffers, as before
    assert per_process(["write"], 1024, 0) == pytest.approx(1024 * 1.2)
    assert per_process(["comprehensive"], 1024, 0) == pytest.approx(2 * 1024 * 1.2)
    # one figure per pass's mode
    assert per_process(["write", "read", "index"], 1024, 10) == pytest.approx(
        1024 * 1.2 + 10 * (result["write"] + result["read"] + result["index"]))
    assert per_process(["write", "read"], 1024, 10, timeline=True) == pytest.approx(
        1024 * 1.2 + 10 * (result["write"] + timeline["write"] + result["read"] + timeline["read"]))
    # benchmark performs no ops
    assert per_process(["benchmark"], 1024, 10) == pytest.approx(1024 * 1.2)
    # meta holds no data, but no longer estimates zero
    assert per_process(["meta"], 1024, 10) == pytest.approx(30 * result["meta"])


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


def spy_worker_args(iotest_mod, monkeypatch):
    # the arguments of every plain worker started, by pass: (mode, counter name, recreates)
    seen = []
    real = iotest_mod.Process

    def process(target, args):
        if target is iotest_mod.worker:
            seen.append((args[4], args[5], args[10]))
        return real(target=target, args=args)

    monkeypatch.setattr(iotest_mod, "Process", process)
    return seen


def read_recreates_by_worker(seen):
    return [recreates for mode, _, recreates in seen if mode == "read"]


@pytest.mark.parametrize(
    "modes,recreates",
    [
        # a comprehensive pass renames and deletes every file it writes
        (["comprehensive", "read"], True),
        (["write", "comprehensive", "read"], True),
        # the last pass before the read decides
        (["comprehensive", "write", "read"], False),
        (["write", "read"], False),
    ],
)
def test_read_after_comprehensive_prepares_content_before_the_barrier(run_main, iotest_mod, work_dir, monkeypatch,
                                                                       modes, recreates):
    seen = spy_worker_args(iotest_mod, monkeypatch)
    result = run_main(modes, process_count=2, file_count=2)
    assert iotest_mod.total_failed_ops(result) == 0
    assert read_recreates_by_worker(seen) == [recreates, recreates]
    assert len(list(work_dir.rglob("temp_*.bin"))) == 4


def test_only_workers_with_missing_or_wrong_size_files_recreate(run_main, iotest_mod, work_dir, monkeypatch):
    run_main(["write"], process_count=4, file_count=2)
    (work_dir / "0001" / "temp_2.bin").unlink()
    (work_dir / "0003" / "temp_1.bin").write_bytes(b"short")
    seen = spy_worker_args(iotest_mod, monkeypatch)
    result = run_main(["read"], process_count=4, file_count=2)
    assert iotest_mod.total_failed_ops(result) == 0
    assert read_recreates_by_worker(seen) == [False, True, False, True]


def test_memory_estimate_counts_a_second_buffer_only_for_recreating_workers(run_main, iotest_mod, work_dir,
                                                                             monkeypatch, rec_tl):
    run_main(["write"], process_count=3)
    (work_dir / "0001" / "temp_1.bin").unlink()
    read_only = iotest_mod.memory_per_process(["read"], 1024, 1)
    recreating = iotest_mod.memory_per_process(["read"], 1024, 1, read_recreates=True)
    estimate = 2 * read_only + recreating
    # just the estimate in RAM: under 90 % of it would warn (and pause, skipped here) without cutting the count
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (int(estimate) + 1, 0))
    result = run_main(["read"], process_count=3, tl=rec_tl)
    assert result["process_count"] == 3
    assert f"Estimated total memory usage: {iotest_mod.format_bytes(estimate)}B" in rec_tl.printed


def test_memory_cut_counts_the_first_workers_that_fit(run_main, iotest_mod, work_dir, monkeypatch):
    # worker 0000's files are missing: it needs room for two buffers, the others for one
    run_main(["write"], process_count=4)
    (work_dir / "0000" / "temp_1.bin").unlink()
    read_only = iotest_mod.memory_per_process(["read"], 1024, 1)
    recreating = iotest_mod.memory_per_process(["read"], 1024, 1, read_recreates=True)
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (int(recreating + 2 * read_only) + 1, 0))
    assert run_main(["read"], process_count=4)["process_count"] == 3


def test_fitting_process_count(iotest_mod):
    fit = iotest_mod.fitting_process_count
    assert fit([2, 1, 1, 1], 4) == 3
    assert fit([2, 1, 1, 1], 5) == 4
    assert fit([2, 1, 1, 1], 1) == 1
    assert fit([3, 3, 3], 8.9) == 2


def test_startup_stats_every_worker_file_once(run_main, iotest_mod, work_dir, monkeypatch):
    # isfile + getsize for the recreate check and two more for the free-space check used to be four per file,
    # and a memory cut walked them again
    run_main(["write"], process_count=3, file_count=4)
    stats = []
    real = os.stat

    def spy(path, *args, **kwargs):
        if isinstance(path, str) and os.path.basename(path).startswith("temp_"):
            stats.append(path)
        return real(path, *args, **kwargs)

    per_worker = iotest_mod.memory_per_process(["read", "write"], 1024, 4)
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (int(2 * per_worker) + 1, 0))
    monkeypatch.setattr(iotest_mod.os, "stat", spy)
    result = run_main(["read", "write"], process_count=3, file_count=4)
    monkeypatch.setattr(iotest_mod.os, "stat", real)
    assert result["process_count"] == 2
    assert len(stats) == 12
    assert len(set(stats)) == 12


def test_scan_worker_files(iotest_mod, work_dir):
    dirs = [str(work_dir / "a"), str(work_dir / "b")]
    for _, _, path in iotest_mod.worker_files(dirs, 3, 2):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * 100)
    os.remove(iotest_mod.worker_file_path(dirs[1], "0001", 0, 2))
    with open(iotest_mod.worker_file_path(dirs[0], "0002", 1, 2), "wb") as f:
        f.write(b"x" * 30)
    # a directory where a file belongs holds no bytes and is not the file
    os.makedirs(iotest_mod.worker_file_path(dirs[1], "0003", 0, 2))
    assert iotest_mod.scan_worker_files(dirs, 4, 2, 100) == [(200, 0), (100, 1), (130, 1), (0, 2)]


def test_every_mode_has_its_buffer_count(iotest_mod):
    assert set(iotest_mod.BUFFER_COPIES) == set(iotest_mod.OPS_PER_FILE) | {"benchmark"}


def test_index_and_meta_runs_reserve_no_buffer_memory(iotest_mod):
    result = iotest_mod.RESULT_BYTES_PER_OP
    for modes in (["index"], ["meta"], ["index", "meta"]):
        ops = sum(iotest_mod.OPS_PER_FILE[mode] * result[mode] for mode in modes)
        # only the per-op figures, however large the files would be
        assert iotest_mod.memory_per_process(modes, 2 ** 30, 10) == pytest.approx(10 * ops)


def test_index_only_run_is_not_cut_for_buffers_it_never_holds(run_main, iotest_mod, monkeypatch):
    # 4 workers' results fit; one 64 MiB buffer would not
    per_process = iotest_mod.memory_per_process(["index"], 64 * MiB, 10)
    assert per_process < 64 * MiB
    monkeypatch.setattr(iotest_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(iotest_mod, "read_meminfo", lambda: (int(4 * per_process) + 1, 0))
    result = run_main(["index"], process_count=4, file_count=10, file_size=64 * MiB)
    assert result["process_count"] == 4
