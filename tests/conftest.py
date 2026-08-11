import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import iotest as iotest_module  # noqa: E402
try:
    import Tee_Logger  # noqa: E402
except ImportError:
    Tee_Logger = iotest_module.Tee_Logger  # noqa: E402


@pytest.fixture
def iotest_mod():
    return iotest_module


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


@pytest.fixture
def make_tl(tmp_path):
    def _make(**kwargs):
        kwargs.setdefault("systemLogFileDir", str(tmp_path / "logs"))
        kwargs.setdefault("programName", "iotest_test")
        kwargs.setdefault("noLog", True)
        kwargs.setdefault("suppressPrintout", True)
        return Tee_Logger.teeLogger(**kwargs)

    return _make


@pytest.fixture
def tiny():
    return {
        "file_size": 1024,
        "file_count": 1,
        "process_count": 1,
        "zeros": True,
        "quiet": True,
        "benchmarkTime": 1,
    }


@pytest.fixture
def run_main(iotest_mod, make_tl, work_dir, tiny):
    def _run(modes, **overrides):
        params = {
            "file_size": tiny["file_size"],
            "file_count": tiny["file_count"],
            "process_count": tiny["process_count"],
            "directory": str(work_dir),
            "modes": modes,
            "quiet": tiny["quiet"],
            "zeros": tiny["zeros"],
            "tl": make_tl(),
            "no_report": True,
            "benchmarkTime": tiny["benchmarkTime"],
        }
        params.update(overrides)
        if "directory" in overrides and not isinstance(params["directory"], (str, list)):
            params["directory"] = str(params["directory"])
        return iotest_mod.main(**params)

    return _run
