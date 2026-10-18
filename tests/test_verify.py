import json
import struct
import zlib

import pytest

from test_cli import run_iotest


def stored_crc(data):
    return struct.unpack("<I", data[-4:])[0]


def computed_crc(data):
    return zlib.crc32(data[:-4]) & 0xffffffff


def mismatch(path, data):
    return (f"verify failed: {path}: CRC-32 mismatch "
            f"(stored 0x{stored_crc(data):08x}, computed 0x{computed_crc(data):08x})")


def flip(path, offset):
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))
    return bytes(data)


def blocks(data, size=512):
    return {data[o:o + size] for o in range(0, len(data), size)}


def worker_files(work_dir):
    return [(work_dir / "0" / name).read_bytes() for name in ("temp_1.bin", "temp_2.bin")]


def test_verify_implies_unique_and_survives_replace(iotest_mod):
    options = iotest_mod.RunOptions(file_size=8, file_count=1, process_count=1, modes=["write"], verify=True)
    # tuning runs every attempt through replace(), which rebuilds the options from their attributes
    for opts in (options, options.replace(process_count=2)):
        assert opts.verify is True and opts.unique is True
    plain = iotest_mod.RunOptions(file_size=8, file_count=1, process_count=1, modes=["write"], unique=True)
    assert plain.verify is False


# 2498: the trailer spans the last two chunks; 3 and 1: blocks smaller than the trailer
@pytest.mark.parametrize("block_size", [None, 1000, 2498, 3, 1])
def test_the_trailer_is_the_crc_of_the_bytes_before_it(iotest_mod, tmp_path, block_size):
    content = iotest_mod.FileContent(2500, False, True, 1, block_size=block_size, verify=True)
    target = tmp_path / "f.bin"
    # the reused buffer still holds the first file's trailer when the second is written
    for index in range(2):
        op = iotest_mod.op_time(content.write(str(target), index, quiet=True))
        assert not isinstance(op, str), op
        data = target.read_bytes()
        assert len(data) == 2500
        assert stored_crc(data) == computed_crc(data)
        op = iotest_mod.op_time(iotest_mod.read_file(str(target), None, 2500, quiet=True, block_size=block_size,
            verify=True))
        assert not isinstance(op, str), op


@pytest.mark.parametrize("block_size", [None, 300])
@pytest.mark.parametrize("modes", [["write", "read"], ["read"], ["random"], ["comprehensive"]])
def test_every_mode_writes_and_checks_the_trailer(run_main, iotest_mod, work_dir, modes, block_size):
    # ["read"] on an empty directory recreates its files before reading them
    result = run_main(modes, process_count=2, file_count=2, file_size=1024, zeros=False, verify=True,
        block_size=block_size)
    assert iotest_mod.total_failed_ops(result) == 0
    for path in work_dir.glob("*/*.bin"):
        data = path.read_bytes()
        assert len(data) == 1024 and stored_crc(data) == computed_crc(data), path


@pytest.mark.parametrize("block_size", [None, 300])
def test_simultaneous_writers_and_layout_write_and_check_the_trailer(run_main, iotest_mod, work_dir, block_size):
    result = run_main(["simultaneous"], process_count=2, file_count=2, file_size=1024, verify=True,
        block_size=block_size, simultaneous_run_id="t", simultaneous_skip_cleanup=True)
    assert iotest_mod.total_failed_ops(result) == 0
    files = list((work_dir / ".iotest_simul_t" / "worker_0").iterdir())
    assert sorted(p.name for p in files) == ["read_1.bin", "read_2.bin", "write_1.bin", "write_2.bin"]
    for path in files:
        data = path.read_bytes()
        assert stored_crc(data) == computed_crc(data), path


@pytest.mark.parametrize("block_size", [None, 1000])
# a data byte at a chunk start, one mid-file, and one inside the trailer
@pytest.mark.parametrize("position", [0, 1500, 2498])
def test_a_flipped_byte_fails_a_later_verified_read(run_main, iotest_mod, work_dir, block_size, position):
    run_main(["write"], file_size=2500, verify=True, block_size=block_size)
    path = work_dir / "0" / "temp_1.bin"
    data = flip(path, position)
    result = run_main(["read"], file_size=2500, verify=True, block_size=block_size)
    assert result["outResults"]["read"] == []
    assert result["failedOps"]["read"] == [mismatch(path, data)]
    assert iotest_mod.total_failed_ops(result) == 1


def test_files_written_without_verify_fail_it(run_main, work_dir):
    run_main(["write"], file_size=2500)
    path = work_dir / "0" / "temp_1.bin"
    result = run_main(["read"], file_size=2500, verify=True)
    assert result["failedOps"]["read"] == [mismatch(path, path.read_bytes())]


def test_verify_alone_stamps_blocks_as_unique_does(run_main, work_dir):
    # with -z and no stamps, every 512-byte block would be the same zeros
    run_main(["write"], file_count=2, file_size=8192, zeros=True, verify=True)
    first, second = worker_files(work_dir)
    assert len(blocks(first)) == len(blocks(second)) == 16
    assert blocks(first).isdisjoint(blocks(second))


def tick_on_crc(iotest_mod, monkeypatch):
    clock = iter(range(100, 200))
    monkeypatch.setattr(iotest_mod.time, "perf_counter", lambda: next(clock))
    marks = []
    real_add = iotest_mod.CrcTrailer.add

    def add(self, chunk, offset):
        marks.append(next(clock))
        return real_add(self, chunk, offset)

    monkeypatch.setattr(iotest_mod.CrcTrailer, "add", add)
    return marks


def test_whole_file_crc_stays_outside_the_timed_write_and_read(iotest_mod, tmp_path, monkeypatch):
    marks = tick_on_crc(iotest_mod, monkeypatch)
    target = str(tmp_path / "f.bin")
    content = iotest_mod.FileContent(2500, True, True, 1, verify=True)
    assert content.write(target, 0, quiet=True) == (101, 102)
    assert marks == [100]
    marks.clear()
    assert iotest_mod.read_file(target, None, 2500, quiet=True, verify=True) == (103, 104)
    assert marks == [105]


def test_chunked_crc_is_inside_the_timed_write_and_read(iotest_mod, tmp_path, monkeypatch):
    marks = tick_on_crc(iotest_mod, monkeypatch)
    target = str(tmp_path / "f.bin")
    content = iotest_mod.FileContent(2500, True, True, 1, block_size=1000, verify=True)
    start, end = content.write(target, 0, quiet=True)
    assert len(marks) == 3 and all(start < tick < end for tick in marks)
    marks.clear()
    start, end = iotest_mod.read_file(target, None, 2500, quiet=True, block_size=1000, verify=True)
    assert len(marks) == 3 and all(start < tick < end for tick in marks)


def test_simultaneous_tuning_lays_out_with_verify(iotest_mod, work_dir, rec_tl, monkeypatch):
    class Stop(Exception):
        pass

    seen = []

    def spy(*args):
        seen.append(args[-1])
        raise Stop

    monkeypatch.setattr(iotest_mod, "run_simultaneous_layout", spy)
    options = iotest_mod.RunOptions(file_size=8192, file_count=1, process_count=2, directories=str(work_dir),
        modes=["simultaneous"], no_report=True, benchmark_time=1, verify=True)
    with pytest.raises(Stop):
        iotest_mod.tune_for_targets(options, rec_tl, 1, None)
    assert seen == [True]


def test_cli_verify_catches_a_flipped_byte(tmp_path):
    work = tmp_path / "w"
    out = tmp_path / "out.json"
    common = ["-fs", "4k", "-fc", "1", "-pc", "1", "-d", str(work), "-nl", "-nr", "-bt", "1", "--verify"]
    proc = run_iotest(common + ["-j", str(out), "write"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    options = json.loads(out.read_text())["options"]
    assert options["verify"] is True and options["unique"] is True
    path = work / "0" / "temp_1.bin"
    data = flip(path, 100)
    proc = run_iotest(common + ["read"])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert mismatch(path, data) in proc.stdout


@pytest.mark.parametrize("size,code", [("4b", 2), ("7b", 2), ("8b", 0)])
def test_cli_verify_needs_a_file_size_of_at_least_8_bytes(tmp_path, size, code):
    work = tmp_path / "w"
    proc = run_iotest(["-fs", size, "-fc", "1", "-pc", "1", "-d", str(work), "-nl", "-nr", "-bt", "1", "--verify",
                       "write"])
    assert proc.returncode == code, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    if code == 2:
        # argparse's usage error, before anything runs
        assert "--verify" in proc.stderr and not work.exists()
    else:
        data = (work / "0" / "temp_1.bin").read_bytes()
        assert len(data) == 8 and stored_crc(data) == computed_crc(data)
