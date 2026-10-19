import os
import tracemalloc

import pytest

import iotest as iotest_module


def test_fadvise_runs_after_fsync_and_not_before_the_write(iotest_mod, tmp_path, monkeypatch):
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("no os.posix_fadvise here")
    calls = []
    real_fsync = os.fsync
    real_writev = os.writev

    def spy_fsync(fd):
        calls.append("fsync")
        real_fsync(fd)

    def spy_writev(fd, bufs):
        calls.append("write")
        return real_writev(fd, bufs)

    monkeypatch.setattr(iotest_mod.os, "fsync", spy_fsync)
    monkeypatch.setattr(iotest_mod.os, "writev", spy_writev)
    monkeypatch.setattr(iotest_mod.os, "posix_fadvise", lambda *a: calls.append("fadvise"))
    start, end = iotest_mod.create_file(str(tmp_path / "f.bin"), b"\x01" * 4096, 4096, quiet=True)
    assert end > start
    assert calls == ["write", "fsync", "fadvise"]


def test_fadvise_is_not_part_of_the_timed_write(iotest_mod, tmp_path, monkeypatch):
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("no os.posix_fadvise here")
    clock = iter(range(100, 200))
    monkeypatch.setattr(iotest_mod.time, "perf_counter", lambda: next(clock))
    marks = []
    monkeypatch.setattr(iotest_mod.os, "posix_fadvise", lambda *a: marks.append(next(clock)))
    start, end = iotest_mod.create_file(str(tmp_path / "f.bin"), b"\x01" * 64, 64, quiet=True)
    # the clock ticks once at start and once at end; fadvise ticks after both
    assert (start, end) == (100, 101)
    assert marks == [102]


@pytest.mark.skipif(iotest_module.fcntl is None, reason="fcntl is not available on Windows")
@pytest.mark.parametrize("op", ["write", "read"])
def test_f_nocache_is_set_when_the_constant_exists(iotest_mod, tmp_path, monkeypatch, op):
    target = tmp_path / "f.bin"
    target.write_bytes(b"\x01" * 64)
    seen = []
    monkeypatch.setattr(iotest_mod.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(iotest_mod.fcntl, "fcntl", lambda fd, cmd, arg=0: seen.append((cmd, arg)))
    if op == "write":
        iotest_mod.create_file(str(target), b"\x01" * 64, 64, quiet=True)
    else:
        iotest_mod.read_file(str(target), b"\x01" * 64, 64, quiet=True)
    assert seen == [(48, 1)]


@pytest.mark.skipif(iotest_module.fcntl is None, reason="fcntl is not available on Windows")
def test_f_nocache_is_skipped_without_the_constant(iotest_mod, tmp_path, monkeypatch):
    seen = []
    if hasattr(iotest_mod.fcntl, "F_NOCACHE"):
        monkeypatch.delattr(iotest_mod.fcntl, "F_NOCACHE")
    monkeypatch.setattr(iotest_mod.fcntl, "fcntl", lambda *a: seen.append(a))
    iotest_mod.create_file(str(tmp_path / "f.bin"), b"\x01" * 64, 64, quiet=True)
    assert seen == []


def test_late_worker_is_recorded(iotest_mod, work_dir):
    late = []
    start = iotest_mod.time.perf_counter() - 5
    iotest_mod.worker(1, 1024, str(work_dir), [], "write", "0", True, True, start, False, False, late)
    assert len(late) == 1 and late[0] >= 5


def test_punctual_worker_is_not_recorded(iotest_mod, work_dir):
    late = []
    start = iotest_mod.time.perf_counter() + 0.3
    iotest_mod.worker(1, 1024, str(work_dir), [], "write", "0", True, True, start, False, False, late)
    assert late == []


def make_late(iotest_mod, monkeypatch, seconds=5):
    real_worker = iotest_mod.worker

    def late_worker(*args):
        args = list(args)
        args[8] -= seconds
        return real_worker(*args)

    monkeypatch.setattr(iotest_mod, "worker", late_worker)


@pytest.mark.patches_workers
def test_late_start_is_reported_and_warned(run_main, rec_tl, iotest_mod, monkeypatch):
    make_late(iotest_mod, monkeypatch)
    result = run_main(["write"], tl=rec_tl, process_count=2)
    n, longest = result["late_starts"]["write"]
    assert n == 2 and longest > 2
    line = [m for m in rec_tl.printed if "Late starts:" in m]
    assert len(line) == 1
    assert "Late starts:           \t2 of 2 workers, up to " in line[0]
    assert " s late (sync figures include the delay)" in line[0]
    assert len([m for m in rec_tl.logged if "started late" in m]) == 1


@pytest.mark.patches_workers
def test_late_start_line_in_every_section_of_the_pass(run_main, rec_tl, iotest_mod, monkeypatch):
    make_late(iotest_mod, monkeypatch)
    result = run_main(["comprehensive"], tl=rec_tl)
    assert set(result["late_starts"]) == {"write", "move", "stat", "read"}
    assert "\n".join(rec_tl.printed).count("Late starts:") == 4
    assert len([m for m in rec_tl.logged if "started late" in m]) == 1


def test_no_late_line_when_nobody_is_late(run_main, rec_tl):
    result = run_main(["write"], tl=rec_tl)
    assert result["late_starts"] == {"write": (0, 0.0)}
    assert not any("Late starts" in m for m in rec_tl.messages)


@pytest.mark.patches_workers
def test_late_starts_follow_renamed_keys(run_main, iotest_mod, monkeypatch):
    make_late(iotest_mod, monkeypatch)
    result = run_main(["write", "write"])
    assert set(result["late_starts"]) == set(result["outResults"]) == {"write_1", "write_2"}


@pytest.mark.patches_workers
def test_random_mode_reports_reads_and_writes_separately(run_main, rec_tl, iotest_mod, monkeypatch):
    # both kinds must occur, so alternate instead of relying on luck
    flips = iter([1, 0, 1, 0])
    monkeypatch.setattr(iotest_mod.random, "getrandbits", lambda n: next(flips))
    result = run_main(["random"], tl=rec_tl, file_count=4)
    assert set(result["outResults"]) == {"random_write", "random_read"}
    assert len(result["outResults"]["random_write"]) == 2
    assert len(result["outResults"]["random_read"]) == 2
    text = "\n".join(rec_tl.printed)
    assert "Report for random_write mode:" in text
    assert "random_read bandwidth (sync):" in text
    assert "random_write bandwidth (call):" in text
    assert result["totalTime"]["random_write"] == result["totalTime"]["random_read"]
    assert result["mode_process_counts"] == {"random_write": 1, "random_read": 1}


@pytest.mark.patches_workers
def test_random_mode_only_lists_kinds_that_occurred(run_main, iotest_mod, monkeypatch):
    monkeypatch.setattr(iotest_mod.random, "getrandbits", lambda n: 1)
    result = run_main(["random"], file_count=2)
    assert list(result["outResults"]) == ["random_write"]


@pytest.mark.patches_workers
def test_random_mode_csv_columns_follow_the_keys(run_main, iotest_mod, work_dir, monkeypatch):
    flips = iter([1, 0, 1, 0])
    monkeypatch.setattr(iotest_mod.random, "getrandbits", lambda n: next(flips))
    run_main(["random"], file_count=4, no_report=False)
    csvs = [p for p in work_dir.glob("*.csv") if not p.name.endswith("_ops.csv")]
    assert len(csvs) == 1
    assert csvs[0].read_text().splitlines()[0] == "random_write_time,random_read_time"


@pytest.mark.patches_workers
def test_random_mode_dead_worker_failures_go_under_random(run_main, iotest_mod, monkeypatch):
    def dying_worker(*args):
        os._exit(3)

    monkeypatch.setattr(iotest_mod, "worker", dying_worker)
    result = run_main(["random"], file_count=2)
    assert list(result["outResults"]) == ["random"]
    assert len(result["failedOps"]["random"]) == 2
    assert result["late_starts"] == {"random": (0, 0.0)}


def test_comprehensive_sections_carry_the_estimate_note(run_main, rec_tl):
    result = run_main(["comprehensive"], tl=rec_tl)
    assert result["sync_estimated"] == {"write": True, "move": True, "stat": True, "read": True}
    text = "\n".join(rec_tl.printed)
    note = "Note: sync figures are estimates: comprehensive mode splits the pass time across write/move/stat/read by their share of op time."
    assert text.count(note) == 4
    assert note + "\n" + "-" * 80 in text


def test_plain_write_pass_has_no_estimate_note(run_main, rec_tl):
    result = run_main(["write"], tl=rec_tl)
    assert result["sync_estimated"] == {"write": False}
    assert not any("sync figures are estimates" in m for m in rec_tl.messages)


MiB = 2 ** 20


def test_read_buffer_alloc_time(iotest_mod):
    alloc = iotest_mod.read_buffer_alloc_time
    margin = iotest_mod.ALLOC_MARGIN
    # one worker: its buffer at the measured rate, plus the margin
    assert alloc(2 ** 30, 1, 2 ** 30) == pytest.approx(1 * margin)
    # parallel workers share the memory bandwidth: together at most twice as fast as one
    assert alloc(2 ** 30, 2, 2 ** 30) == pytest.approx(1 * margin)
    assert alloc(2 ** 30, 8, 2 ** 30) == pytest.approx(4 * margin)
    assert alloc(512 * MiB, 8, 2 ** 30) == pytest.approx(2 * margin)
    # no worker allocates one, or no rate could be measured
    assert alloc(2 ** 30, 0, 2 ** 30) == 0
    assert alloc(2 ** 30, 8, None) == 0


def test_process_start_delay(iotest_mod):
    assert iotest_mod.process_start_delay(36) == pytest.approx(1.18)
    assert iotest_mod.process_start_delay(36, 0.5, 2.0) == pytest.approx(3.68)


@pytest.mark.parametrize(
    "modes,write_count,workers",
    [
        (["read"], None, 6),
        (["random"], None, 6),
        (["comprehensive"], None, 6),
        (["write", "read", "index"], None, 6),
        # simultaneous: only its readers
        (["simultaneous"], None, 3),
        (["simultaneous"], 4, 2),
        (["write"], None, 0),
        (["index", "meta"], None, 0),
    ],
)
def test_read_buffer_workers(iotest_mod, modes, write_count, workers):
    assert iotest_mod.read_buffer_workers(modes, 6, write_count) == workers


def test_alloc_speed_sample_is_at_most_64_mib(iotest_mod):
    tracemalloc.start()
    try:
        speed = iotest_mod.measure_alloc_speed(8 * 2 ** 30)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert speed > 0
    assert 64 * MiB <= peak < 65 * MiB


@pytest.mark.parametrize("prepared", [True, False])
def test_start_delay_covers_read_buffer_allocation(run_main, iotest_mod, monkeypatch, rec_tl, prepared):
    # 100 KiB/s: 2 workers x 1 KiB buffers take 2 KiB / (2 x 100 KiB/s) = 0.01 s, plus the margin. Read-only
    # runs with every file in place skip the generation benchmark, and the delay was 0.005 x pc + 1 s
    if prepared:
        run_main(["write"], process_count=2)
    sizes = []
    monkeypatch.setattr(iotest_mod, "measure_alloc_speed", lambda size: sizes.append(size) or 100 * 1024)
    run_main(["read"], process_count=2, tl=rec_tl)
    assert sizes == [1024]
    alloc = 0.01 * iotest_mod.ALLOC_MARGIN
    assert f"Read buffer alloc time:\t{alloc:.4f} s" in rec_tl.printed
    delays = [line for line in rec_tl.printed if line.startswith("Process start delay:")]
    (delay,) = delays
    gen = 0.0
    if not prepared:
        # the workers recreate the files, so they also generate content first
        (gen_line,) = [line for line in rec_tl.printed if line.startswith("Gen time calculated:")]
        gen = float(gen_line.split("\t")[1].split()[0])
    assert float(delay.split("\t")[1].split()[0]) == pytest.approx(0.005 * 2 + 1 + gen + alloc, abs=2e-4)


def test_write_runs_add_no_read_buffer_allocation(run_main, iotest_mod, monkeypatch, rec_tl):
    monkeypatch.setattr(iotest_mod, "measure_alloc_speed", lambda size: pytest.fail("no read buffers to time"))
    run_main(["write", "index"], tl=rec_tl)
    assert not any(line.startswith("Read buffer alloc time") for line in rec_tl.printed)
    assert sum(line.startswith("Process start delay") for line in rec_tl.printed) == 1


def test_simultaneous_reader_allocates_its_buffer_before_the_start_barrier(iotest_mod, work_dir, rec_tl, monkeypatch):
    iotest_mod.run_simultaneous_layout(1, 4096, [str(work_dir)], "t", 1, True, True, rec_tl)
    made_at = []
    real = bytearray

    def spy(*args):
        if args == (4096,):
            made_at.append(iotest_mod.time.perf_counter())
        return real(*args)

    monkeypatch.setattr(iotest_mod, "bytearray", spy, raising=False)
    start = iotest_mod.time.perf_counter() + 0.3
    results = []
    iotest_mod.simultaneous_role_worker("read", 1, 4096, str(work_dir), "t", iotest_mod.worker_counter_name(0),
        results, True, True, start)
    assert len(made_at) == 1 and made_at[0] < start
    assert not [s for s in iotest_mod.result_samples(results) if isinstance(s, str)]
