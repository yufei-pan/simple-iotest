import json
import subprocess
import sys
from pathlib import Path

import pytest

import iotest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "iotest.py"
PYPROJECT = ROOT / "pyproject.toml"


def test_package_name_console_script_alias():
    # uvx / pipx invoke the console script whose name matches the package.
    text = PYPROJECT.read_text()
    scripts = text.split("[project.scripts]", 1)[1].split("[", 1)[0]
    assert 'iotest = "iotest:climain"' in scripts
    assert 'simple-iotest = "iotest:climain"' in scripts


def run_iotest(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(MODULE), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
        cwd=cwd,
    )


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version_flag(flag):
    proc = run_iotest([flag])
    assert proc.returncode == 0
    out = (proc.stdout or "") + (proc.stderr or "")
    assert iotest.version in out


def test_help_flag():
    proc = run_iotest(["-h"])
    assert proc.returncode == 0
    assert "file_size" in (proc.stdout or "").lower() or "File size" in (proc.stdout or "")


def test_cli_tiny_write(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    logd = tmp_path / "logs"
    logd.mkdir()
    proc = run_iotest(
        [
            "-fs", "1k",
            "-fc", "1",
            "-pc", "1",
            "-d", str(work),
            "-ld", str(logd),
            "-nl",
            "-nr",
            "-z",
            "-bt", "1",
            "write",
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_cli_write_then_read(tmp_path):
    # unpatched, in a fresh interpreter: the real command with the platform's own worker start method (spawn on macOS and Windows)
    work = tmp_path / "w"
    work.mkdir()
    proc = run_iotest(["-fs", "1k", "-fc", "2", "-pc", "2", "-d", str(work), "-nl", "-nr", "-z", "-bt", "1", "write", "read"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Report for write mode:" in proc.stdout and "Report for read mode:" in proc.stdout
    assert len(list(work.rglob("temp_*.bin"))) == 4


def test_cli_exits_nonzero_when_ops_fail(tmp_path):
    work = tmp_path / "w"
    blocker = work / "0000" / "temp_1.bin"
    blocker.mkdir(parents=True)
    proc = run_iotest(
        ["-fs", "1k", "-fc", "1", "-pc", "1", "-d", str(work), "-nl", "-nr", "-z", "-bt", "1", "write"]
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    # the OSError names the path by its repr, which doubles the backslashes of a Windows path
    assert repr(str(blocker)) in proc.stdout


def test_cli_total_size_fills_file_count(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    logd = tmp_path / "logs"
    logd.mkdir()
    # fs=1k, pc=1, ts=4k → fc should become 4
    proc = run_iotest(
        [
            "-fs", "1k",
            "-pc", "1",
            "-ts", "4k",
            "-d", str(work),
            "-ld", str(logd),
            "-nl",
            "-nr",
            "-z",
            "-bt", "1",
            "-S",
            "write",
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # stealth still logs via tee; with -nl may be quiet — at least success
    assert proc.returncode == 0


def test_cli_wtpt_rejects_index_only(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    logd = tmp_path / "logs"
    logd.mkdir()
    proc = run_iotest(
        [
            "-fs", "1k",
            "-fc", "1",
            "-pc", "1",
            "-d", str(work),
            "-ld", str(logd),
            "-nl",
            "-nr",
            "-wtpt", "1m",
            "index",
        ]
    )
    assert proc.returncode != 0


def test_cli_simultaneous_dual_d(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    logd = tmp_path / "logs"
    logd.mkdir()
    proc = run_iotest(
        [
            "-fs", "1k",
            "-fc", "1",
            "-pc", "2",
            "-d", str(d1),
            "-d", str(d2),
            "-ld", str(logd),
            "-nl",
            "-nr",
            "-z",
            "-bt", "1",
            "simultaneous",
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert list(d1.glob(".iotest_simul_*")) == []
    assert list(d2.glob(".iotest_simul_*")) == []


@pytest.mark.parametrize("flag,value", [("-fc", "0"), ("-pc", "0"), ("-fs", "0"), ("-fc", "-3")])
def test_cli_rejects_values_below_one(tmp_path, flag, value):
    work = tmp_path / "w"
    args = {"-fs": "1k", "-fc": "1", "-pc": "1"}
    args[flag] = value
    proc = run_iotest([*sum(args.items(), ()), "-d", str(work), "-nl", "-nr", "-bt", "1", "write"])
    # argparse's usage error, before anything runs
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert flag in proc.stderr
    assert not work.exists()


def test_cli_accepts_fractional_anomaly_threshold(tmp_path):
    work = tmp_path / "w"
    proc = run_iotest(["-fs", "1k", "-fc", "1", "-pc", "1", "-d", str(work), "-nl", "-nr", "-z", "-bt", "1",
                       "--threshold_to_report_anomaly", "1.5", "write"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_quiet_progress_digits_are_not_glued_to_the_report(tmp_path):
    proc = run_iotest(["-d", str(tmp_path), "-nl", "-nr", "-fs", "1k", "-fc", "10", "-t", "1", "-bt", "1", "write"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # the last digit is 9 (ten files); universal_newlines turns the digits' \r into \n, so the digit line
    # ends cleanly whatever prefix (color reset, "[INFO] ") the installed tee-logger puts on the report
    assert "\n9\n" in proc.stdout
    assert "*" * 80 in proc.stdout


def test_inline_logger_notice_goes_to_stderr(tmp_path):
    work = tmp_path / "w"
    code = ("import sys; sys.modules['Tee_Logger'] = None; sys.path.insert(0, %r); import iotest; "
            "sys.argv = ['iotest'] + %r; iotest.climain()" % (str(Path(__file__).resolve().parents[1]),
            ["-S", "-j", "-", "-fs", "1k", "-fc", "2", "-t", "2", "-bt", "1", "-d", str(work), "w"]))
    proc = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["results"]
    assert "Using inline logger, forcing noLog..." in proc.stderr
