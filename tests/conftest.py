import json
import subprocess
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


class _RecTeeLogger:
    """Minimal tee-logger double that records every rendering call.

    `main()` and the tuning functions only need teeprint/teeerror/teelog/info
    plus the `currentDateTime` / `logFileName` attributes; nothing here connects
    to real files, terminals, or logs.
    """

    def __init__(self):
        self.printed = []
        self.logged = []
        self.currentDateTime = "rec_tl_run"
        self.logFileName = "rec_tl.log"

    def teeprint(self, msg, *a, **k):
        self.printed.append(str(msg))

    def teeerror(self, msg, *a, **k):
        self.printed.append(str(msg))

    def teelog(self, msg, *a, **k):
        self.logged.append(str(msg))

    def info(self, msg, *a, **k):
        self.logged.append(str(msg))

    @property
    def messages(self):
        return self.printed + self.logged


@pytest.fixture
def rec_tl():
    return _RecTeeLogger()


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


# Runs in its own process: iotest forks workers, and forking while a server
# thread is alive in the test process triggers fork-with-threads warnings.
_WEBHOOK_SERVER = r'''
import http.server, json, sys
status, log_path = int(sys.argv[1]), sys.argv[2]
class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        headers = {k.lower(): v for k, v in self.headers.items()}
        with open(log_path, "a") as f:
            f.write(json.dumps({"headers": headers, "json": json.loads(body)}) + "\n")
        self.send_response(status)
        self.end_headers()
    def log_message(self, *args):
        pass
server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
'''


@pytest.fixture
def webhook(tmp_path):
    """Start a local endpoint that records each POST; returns (url, received)."""
    procs = []

    def _start(status=204):
        log_path = tmp_path / f"webhook_{len(procs)}.jsonl"
        proc = subprocess.Popen(
            [sys.executable, "-c", _WEBHOOK_SERVER, str(status), str(log_path)],
            stdout=subprocess.PIPE, text=True)
        procs.append(proc)
        port = int(proc.stdout.readline())

        def received():
            if not log_path.exists():
                return []
            return [json.loads(line) for line in log_path.read_text().splitlines()]

        return f"http://127.0.0.1:{port}/hook", received

    yield _start
    for proc in procs:
        proc.kill()
        proc.wait()
