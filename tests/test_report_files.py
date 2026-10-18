import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "iotest.py"


def test_report_base_name_keeps_short_names_unchanged(iotest_mod):
    name = iotest_mod.report_base_name(["write"], ["/tmp/a b"], 1024, 2, 3, "20260101")
    assert name == "iotest_write_-tmp-a_b_fs=1024_fc=2_pc=3_20260101"


def test_report_base_name_shortens_only_the_directory_part(iotest_mod):
    dirs = ["/very/long/" + "x" * 80 + str(i) for i in range(5)]
    name = iotest_mod.report_base_name(["write", "read"], dirs, 1024, 2, 3, "20260101")
    longest = max(len(s.encode()) for s in iotest_mod.REPORT_SUFFIXES)
    assert len(name.encode()) + longest <= 255
    assert re.search(r"~[0-9a-f]{8}_fs=1024_fc=2_pc=3_20260101$", name)
    assert name.startswith("iotest_write-read_-very-long-xxx")
    other = iotest_mod.report_base_name(["write", "read"], dirs[:4], 1024, 2, 3, "20260101")
    assert other != name


def test_report_base_name_counts_bytes_not_characters(iotest_mod):
    name = iotest_mod.report_base_name(["write"], ["/" + "é" * 200], 1024, 2, 3, "20260101")
    longest = max(len(s.encode()) for s in iotest_mod.REPORT_SUFFIXES)
    assert len(name.encode()) + longest <= 255
    assert "~" in name


def test_long_directories_still_write_result_files(run_main, rec_tl, tmp_path):
    dirs = []
    for i in range(4):
        d = tmp_path / ("d" * 60 + str(i))
        d.mkdir()
        dirs.append(str(d))
    assert len("_".join(dirs).encode()) > 255
    result = run_main(["write"], directory=dirs, process_count=2, tl=rec_tl, no_report=False, timeline=True)
    assert result["report_errors"] == []
    names = sorted(p.name for p in Path(dirs[0]).glob("iotest_*"))
    assert len(names) == 3
    for name in names:
        assert len(name.encode()) <= 255
        assert re.search(r"~[0-9a-f]{8}_fs=", name)
    assert any("Report for write mode" in line for line in rec_tl.printed)


def test_short_directory_keeps_todays_name(run_main, rec_tl, work_dir):
    run_main(["write"], tl=rec_tl, no_report=False)
    # the same separator rule iotest applies, so Windows paths (C:\...) match too
    dir_part = str(work_dir).replace("/", "-").replace("\\", "-").replace(":", "-").replace("--", "-").replace(" ", "_")
    expected = "iotest_write_%s_fs=1024_fc=1_pc=1_rec_tl_run" % dir_part
    assert (work_dir / (expected + ".csv")).exists()
    assert (work_dir / (expected + "_report.txt")).exists()


def test_failed_result_write_keeps_report_and_continues(run_main, rec_tl, iotest_mod, work_dir):
    base = iotest_mod.report_base_name(["write"], [str(work_dir)], 1024, 1, 1, rec_tl.currentDateTime)
    # a directory in the csv's place makes open() fail without chmod
    os.mkdir(str(work_dir / (base + ".csv")))
    result = run_main(["write"], tl=rec_tl, no_report=False)
    assert len(result["report_errors"]) == 1
    assert result["report_errors"][0].startswith("Could not write ")
    assert (work_dir / (base + "_report.txt")).exists()
    assert any("Report for write mode" in line for line in rec_tl.printed)
    assert result["report_errors"][0] in rec_tl.printed


def test_cli_exits_1_when_result_file_write_fails(iotest_mod, monkeypatch, tmp_path, capsys):
    real_open = open

    def failing_open(path, *args, **kwargs):
        if str(path).endswith(".csv"):
            raise OSError("disk full")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(iotest_mod, "open", failing_open, raising=False)
    monkeypatch.setattr(sys, "argv", ["iotest", "-fs", "1k", "-fc", "1", "-pc", "1", "-d", str(tmp_path / "w"),
                                      "-nl", "-z", "-bt", "1", "write"])
    with pytest.raises(SystemExit) as exc:
        iotest_mod.climain()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Report for write mode" in out
    assert "disk full" in out


def cli(args):
    return subprocess.run([sys.executable, str(MODULE), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, check=False)


@pytest.mark.parametrize("args,needle", [
    (["-fs", "abc"], "argument -fs/--file_size: invalid size value: 'abc'"),
    (["-fs", "10x"], "'10x'"),
    (["-fs", "nan"], "'nan'"),
    (["-fs", "inf"], "'inf'"),
    (["-fs", "0"], "argument -fs/--file_size"),
    (["-fs", "-5"], "argument -fs/--file_size"),
    (["-ts", "bogus"], "argument -ts/--total_size: invalid size value: 'bogus'"),
    (["-ts", "0"], "argument -ts/--total_size"),
    (["-ts", "-1g"], "argument -ts/--total_size"),
    (["-wtpt", "fast"], "argument -wtpt/--write_target_throughput: invalid size value: 'fast'"),
    (["-wtpt", "0"], "argument -wtpt/--write_target_throughput"),
    (["-rtpt", "fast", "read"], "argument -rtpt/--read_target_throughput: invalid size value: 'fast'"),
    (["-rtpt", "-1", "read"], "argument -rtpt/--read_target_throughput"),
    (["-wtpt", "100m", "read"], "-wtpt/--write_target_throughput requires a mode with a write pass"),
    (["-rtpt", "100m", "write"], "-rtpt/--read_target_throughput requires a mode with a read pass"),
])
def test_invalid_arguments_exit_2_without_traceback(tmp_path, args, needle):
    proc = cli([*args, "-d", str(tmp_path / "w"), "-nl", "-nr", "-bt", "1"])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert needle in proc.stderr
    assert "Traceback" not in proc.stdout + proc.stderr


def test_report_is_printed_exactly_once(run_main, rec_tl):
    run_main(["write"], tl=rec_tl, no_report=False)
    assert sum("Report for write mode" in line for line in rec_tl.printed) == 1
