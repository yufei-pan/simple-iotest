import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iotest as iotest_module  # noqa: E402


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


def run_options(iotest_mod, **kwargs):
    kwargs.setdefault("file_size", 1024)
    kwargs.setdefault("file_count", 1)
    kwargs.setdefault("process_count", 1)
    kwargs.setdefault("modes", ["write"])
    return iotest_mod.RunOptions(**kwargs)


def test_run_options_replace_returns_a_new_object(iotest_mod, work_dir):
    options = run_options(iotest_mod, process_count=2, directories=str(work_dir))
    changed = options.replace(process_count=4, quiet=False)
    assert changed is not options
    assert (changed.process_count, changed.quiet) == (4, False)
    assert (options.process_count, options.quiet) == (2, True)
    assert changed.directories == options.directories == [str(work_dir)]


def test_run_options_rejects_unknown_options(iotest_mod):
    with pytest.raises(TypeError):
        run_options(iotest_mod, benchmarkTime=5)
    with pytest.raises(TypeError):
        run_options(iotest_mod).replace(benchmarkTime=5)
    # keyword arguments only
    with pytest.raises(TypeError):
        iotest_mod.RunOptions(1024, 1, 1, ["write"])


def test_run_options_normalizes_directories(iotest_mod, work_dir):
    assert run_options(iotest_mod, directories=str(work_dir)).directories == [str(work_dir)]
    assert run_options(iotest_mod, directories=None).directories == [os.getcwd()]
    assert run_options(iotest_mod).directories == [os.getcwd()]
    assert run_options(iotest_mod, directories=(str(work_dir), "b")).directories == [str(work_dir), "b"]


@pytest.mark.parametrize(
    "version,expected",
    [("6.100", True), ("6.9", False), ("7.0", True), ("6.34", True), ("6.34.1", True)],
)
def test_version_at_least_compares_integer_parts(iotest_mod, version, expected):
    assert iotest_mod.version_at_least(version, "6.34") is expected


@pytest.mark.parametrize("tee_version,used", [("6.100", True), ("6.9", False), ("dev", False)])
def test_tee_logger_version_gate(monkeypatch, tee_version, used):
    # an old or unparseable Tee_Logger falls back to the inline logger
    fake = types.ModuleType("Tee_Logger")
    fake.version = tee_version
    monkeypatch.setitem(sys.modules, "Tee_Logger", fake)
    spec = importlib.util.spec_from_file_location("iotest_gate_check", iotest_module.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.Tee_Logger is fake) is used
    if not used:
        assert module.Tee_Logger.version == "0.1 inline"


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


def test_calc_iops(iotest_mod):
    # 4 ops across 2 processes: 1.0 s of call time per process -> 4 call IOPS
    call_iops, sync_iops = iotest_mod.calc_iops([0.5, 0.5, 0.5, 0.5], 2.0, 2)
    assert call_iops == pytest.approx(4.0)
    assert sync_iops == pytest.approx(2.0)


def test_calc_iops_zero_times_are_none(iotest_mod):
    assert iotest_mod.calc_iops([0.0, 0.0], 0, 1) == (None, None)
    assert iotest_mod.calc_iops([0.1], None, 1) == (pytest.approx(10.0), None)
    assert iotest_mod.calc_iops([0.1], 1.0, 0) == (None, pytest.approx(1.0))


def test_format_iops(iotest_mod):
    assert iotest_mod.format_iops(None) == "N/A"
    assert iotest_mod.format_iops(1234567.891) == "1,234,567.89"
    assert iotest_mod.format_iops(3.5) == "3.50"


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
    iotest_mod.read_file(str(src), len(content), quiet=True)
    iotest_mod.stat_file(str(src), quiet=True)
    iotest_mod.move_file(str(src), str(dst), quiet=True)
    assert not src.exists()
    assert dst.is_file()
    # index_file creates `<name>.index`, stats it, then rmdir — no leftovers
    before = set(tmp_path.iterdir())
    iotest_mod.index_file(str(tmp_path / "idx_target"), quiet=True)
    after = set(tmp_path.iterdir())
    assert after == before


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("1m", 1 * 1024 * 1024),
        ("500m", 500 * 1024 * 1024),
        ("1.5m", int(1.5 * 1024 * 1024)),
        ("2g", 2 * 1024 ** 3),
        ("3k", 3 * 1024),
        ("1t", 1024 ** 4),
        ("64b", 64),
        ("1.5b", 1),
        (" 2G ", 2 * 1024 ** 3),  # whitespace + case-insensitive
        ("30", 30 * 1024 * 1024),  # unitless defaults to MB
    ],
)
def test_parse_size_arg_units(iotest_mod, arg, expected):
    assert iotest_mod.parse_size_arg(arg) == expected


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("30", 30),  # unitless_as_mb=False -> raw bytes
        ("100", 100),
        ("0", 0),
        ("1024", 1024),
    ],
)
def test_parse_size_arg_unitless_raw(iotest_mod, arg, expected):
    assert iotest_mod.parse_size_arg(arg, unitless_as_mb=False) == expected


def test_parse_size_arg_negative_raises_valueerror(iotest_mod):
    with pytest.raises(ValueError):
        iotest_mod.parse_size_arg("not-a-size")
    with pytest.raises(ValueError):
        iotest_mod.parse_size_arg("abc")
    with pytest.raises(ValueError):
        iotest_mod.parse_size_arg("12z34")


@pytest.mark.parametrize(
    "level",
    ["info", "debug", "warning", "error", "critical", "ok", "okgreen", "okblue", "okcyan", "bogus"],
)
def test_print_with_color_emits_message(iotest_mod, capsys, level):
    iotest_mod.printWithColor("hello-print", level)
    captured = capsys.readouterr().out
    assert "hello-print" in captured


@pytest.mark.parametrize(
    "level,marker",
    [
        ("warning", iotest_module.bcolors.warning),
        ("error", iotest_module.bcolors.warning),
        ("critical", iotest_module.bcolors.critical),
        ("ok", iotest_module.bcolors.OKGREEN),
        ("okgreen", iotest_module.bcolors.OKGREEN),
        ("okblue", iotest_module.bcolors.OKBLUE),
        ("okcyan", iotest_module.bcolors.OKCYAN),
        ("debug", iotest_module.bcolors.debug),
        ("bogus", iotest_module.bcolors.info),
    ],
)
def test_print_with_color_ansi_markers(iotest_mod, capsys, level, marker):
    # error uses the warning colour; bogus falls through to the info branch
    iotest_mod.printWithColor("x", level)
    assert marker in capsys.readouterr().out


def test_worker_log_quiet_suppresses(iotest_mod, capsys):
    iotest_mod.worker_log("silenced", quiet=True)
    assert capsys.readouterr().out == ""
    iotest_mod.worker_log("visible", quiet=False)
    assert "visible" in capsys.readouterr().out


def test_worker_color_log_wraps_the_message_in_the_color(iotest_mod, capsys):
    iotest_mod.worker_color_log((1, 2, 3), "Worker 0 finished.")
    assert capsys.readouterr().out == "\033[38;2;1;2;3mWorker 0 finished.\033[0m\n"
    iotest_mod.worker_color_log((1, 2, 3), "silenced", quiet=True)
    assert capsys.readouterr().out == ""
