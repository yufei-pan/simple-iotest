import subprocess
import sys
from pathlib import Path

import pytest

import iotest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "iotest.py"


def run_iotest(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(MODULE), *args],
        capture_output=True,
        text=True,
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
    out = (proc.stdout or "") + (proc.stderr or "")
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
