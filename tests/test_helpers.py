import math
import os

import pytest


def test_version_string(iotest_mod):
    assert isinstance(iotest_mod.version, str)
    assert iotest_mod.version == iotest_mod.__version__


def test_format_bytes_round_trip_1024(iotest_mod):
    n = iotest_mod.format_bytes("1.5 GiB", to_int=True)
    assert n == 1610612736
    s = iotest_mod.format_bytes(n, use_1024_bytes=True, to_str=True)
    assert "Gi" in s


def test_format_bytes_invalid_returns_zero(iotest_mod):
    assert iotest_mod.format_bytes("not-a-size", to_int=True) == 0


@pytest.mark.parametrize(
    "token,expected",
    [
        ("w", ["write"]),
        ("rw", ["write", "read"]),
        ("wr", ["write", "read"]),
        ("rwi", ["write", "read", "index"]),
        ("wri", ["write", "read", "index"]),
        ("c", ["comprehensive"]),
        ("s", ["simultaneous"]),
        ("i", ["index"]),
        ("random", ["random"]),
        ("benchmark", ["benchmark"]),
        ("b", ["benchmark"]),
    ],
)
def test_parse_modes_aliases(iotest_mod, token, expected):
    assert iotest_mod.parse_modes([token]) == expected


def test_parse_modes_unknown_ignored(iotest_mod):
    assert iotest_mod.parse_modes(["nope"]) == []


def test_normalize_directories(iotest_mod, work_dir):
    assert iotest_mod.normalize_directories(str(work_dir)) == [str(work_dir)]
    assert iotest_mod.normalize_directories([str(work_dir), "b"]) == [str(work_dir), "b"]
    assert iotest_mod.normalize_directories(None) == [os.getcwd()]


def test_addToDicWithoutOverwrite(iotest_mod):
    d = {}
    iotest_mod.addToDicWithoutOverwrite(d, "write", [1.0])
    assert d == {"write": [1.0]}
    iotest_mod.addToDicWithoutOverwrite(d, "write", [2.0])
    assert "write_1" in d and d["write_1"] == [1.0]
    assert d["write_2"] == [2.0]


def test_throughput_status(iotest_mod):
    assert iotest_mod.throughput_status(None, 100) == "missing"
    assert iotest_mod.throughput_status(50, 100) == "below"
    assert iotest_mod.throughput_status(100, 100) == "ok"
    assert iotest_mod.throughput_status(100 * iotest_mod.TUNING_TOLERANCE, 100) == "ok"
    assert iotest_mod.throughput_status(100 * iotest_mod.TUNING_TOLERANCE + 1, 100) == "above"


def test_sync_and_measure_bandwidth(iotest_mod):
    out = {"write": [0.1, 0.1]}
    total = {"write": 0.5}
    bps = iotest_mod.sync_bandwidth_bps(out, total, 1024, 2, 2, "write")
    assert bps == pytest.approx((1024 * 2 * 2) / 0.5)
    result = {
        "outResults": out,
        "totalTime": total,
        "file_size": 1024,
        "file_count": 2,
        "process_count": 2,
        "mode_process_counts": {"write": 2},
    }
    assert iotest_mod.measure_role_bandwidth(result, "write") == pytest.approx(bps)
    assert iotest_mod.sync_bandwidth_bps(out, total, 1024, 2, 2, "read") is None


def test_supports_throughput_tuning(iotest_mod):
    assert iotest_mod.supports_write_throughput_tuning(["write"]) is True
    assert iotest_mod.supports_write_throughput_tuning(["index"]) is False
    assert iotest_mod.supports_read_throughput_tuning(["read"]) is True
    assert iotest_mod.supports_read_throughput_tuning(["write"]) is False
    assert iotest_mod.supports_read_throughput_tuning(["simultaneous"]) is True


def test_estimate_simultaneous_split(iotest_mod):
    w, r = iotest_mod.estimate_simultaneous_split(4, 100, 100)
    assert w + r == 4
    assert w >= 1 and r >= 1
    w2, r2 = iotest_mod.estimate_simultaneous_split(4, 300, 100)
    assert w2 + r2 == 4
    assert w2 >= w


def test_almost_urandom_length(iotest_mod):
    buf = iotest_mod.almost_urandom(1000)
    assert isinstance(buf, (bytes, bytearray))
    assert len(buf) == 1000


def test_create_read_move_stat_index(iotest_mod, tmp_path):
    content = b"\x00" * 64
    src = tmp_path / "a.bin"
    dst = tmp_path / "b.bin"
    iotest_mod.create_file(str(src), content, len(content), quiet=True)
    assert src.is_file()
    assert src.stat().st_size == 64
    iotest_mod.read_file(str(src), content, len(content), quiet=True)
    iotest_mod.stat_file(str(src), quiet=True)
    iotest_mod.move_file(str(src), str(dst), quiet=True)
    assert not src.exists()
    assert dst.is_file()
    # index_file creates `<name>.index`, stats it, then rmdir — no leftovers
    before = set(tmp_path.iterdir())
    iotest_mod.index_file(str(tmp_path / "idx_target"), quiet=True)
    after = set(tmp_path.iterdir())
    assert after == before
