import json
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


# on Linux a name that is not UTF-8 reaches Python as surrogate escapes: -d $'/scratch/\xff' is '/scratch/\udcff'
# (os.fsdecode(b"\xff") there; written out because Windows' fsdecode refuses the byte at import)
UNDECODABLE = "\udcff"
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux file names are bytes; others are Unicode")


@linux_only
@pytest.mark.parametrize("repeat", [1, 300])
def test_report_base_name_takes_undecodable_directory_names(iotest_mod, tmp_path, repeat):
    # 300: too long, so the directory part is cut, on whole characters
    name = iotest_mod.report_base_name(["write"], ["/scratch/" + UNDECODABLE * repeat], 1024, 2, 3, "20260101")
    longest = max(len(os.fsencode(s)) for s in iotest_mod.REPORT_SUFFIXES)
    assert len(os.fsencode(name)) + longest <= 255
    assert UNDECODABLE in name
    assert ("~" in name) == (repeat > 1)
    for suffix in iotest_mod.REPORT_SUFFIXES:
        with open(str(tmp_path / (name + suffix)), "w") as f:
            f.write("ok")
    assert len(list(tmp_path.iterdir())) == len(iotest_mod.REPORT_SUFFIXES)


@linux_only
def test_an_undecodable_directory_gets_its_result_files(run_main, rec_tl, tmp_path):
    d = tmp_path / ("data" + UNDECODABLE)
    result = run_main(["write"], directory=str(d), tl=rec_tl, no_report=False, timeline=True)
    assert result["report_errors"] == []
    assert len(list(d.glob("iotest_*"))) == 3


def undecodable_dir_with_a_failing_read(tmp_path):
    # the read must recreate temp_1.bin, which a directory blocks: its error text carries the raw path
    d = tmp_path / ("data" + UNDECODABLE)
    (d / "0000" / "temp_1.bin").mkdir(parents=True)
    return d


def assert_result_files(d, failed_path):
    names = sorted(p.name for p in d.iterdir() if p.is_file())
    assert len([n for n in names if n.endswith("_ops.csv")]) == 1
    assert len([n for n in names if n.endswith("_report.txt")]) == 1
    assert len([n for n in names if n.endswith(".csv") and not n.endswith("_ops.csv")]) == 1
    assert "out.json" in names
    escaped = failed_path.replace(UNDECODABLE, "\\udcff")
    for name in names:
        # valid UTF-8 throughout, the undecodable byte written as \udcff
        text = (d / name).read_bytes().decode("utf-8")
        if name.endswith(("_report.txt", "_ops.csv")):
            assert "recreating missing " + escaped in text, name
    data = json.loads((d / "out.json").read_text())
    assert data["failed_ops"] == 1
    assert data["results"]["read"]["first_error"].startswith("recreating missing " + failed_path)


@linux_only
def test_error_texts_with_undecodable_paths_keep_every_result_file(run_main, rec_tl, iotest_mod, tmp_path):
    d = undecodable_dir_with_a_failing_read(tmp_path)
    result = run_main(["read"], directory=str(d), tl=rec_tl, no_report=False, timeline=True,
        json_path=str(d / "out.json"))
    assert result["report_errors"] == []
    assert iotest_mod.total_failed_ops(result) == 1
    assert_result_files(d, str(d / "0000" / "temp_1.bin"))


@linux_only
@pytest.mark.skipif(sys.version_info < (3, 7), reason="stdout takes undecodable bytes only in UTF-8 mode (3.7+)")
def test_cli_in_an_undecodable_directory_exits_1_for_the_failed_op_only(tmp_path):
    d = undecodable_dir_with_a_failing_read(tmp_path)
    proc = subprocess.run([sys.executable, str(MODULE), "-fs", "1k", "-fc", "1", "-pc", "1", "-d", str(d), "-nl",
                           "-bt", "1", "-tl", "-j", str(d / "out.json"), "read"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                          env=dict(os.environ, PYTHONUTF8="1"))
    out = os.fsdecode(proc.stdout)
    assert proc.returncode == 1, out + os.fsdecode(proc.stderr)
    assert "Could not write" not in out and "Traceback" not in os.fsdecode(proc.stderr)
    assert_result_files(d, str(d / "0000" / "temp_1.bin"))


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


def test_unencodable_report_text_lands_in_report_errors(iotest_mod, rec_tl, tmp_path):
    def write(f):
        raise UnicodeEncodeError("charmap", "x", 0, 1, "character maps to <undefined>")

    errors = []
    assert iotest_mod.write_report_file(str(tmp_path / "r.txt"), write, rec_tl, errors) is False
    assert len(errors) == 1 and "Could not write" in errors[0]
    assert any("Could not write" in line for line in rec_tl.printed)


def test_report_files_are_written_as_utf8(iotest_mod, rec_tl, tmp_path):
    path = tmp_path / "r.txt"
    errors = []
    assert iotest_mod.write_report_file(str(path), lambda f: f.write("path /data/测试"), rec_tl, errors)
    assert errors == []
    assert path.read_bytes() == "path /data/测试".encode("utf-8")
