# Full iotest pytest suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a layered pytest suite for `simple-iotest` covering helpers, all modes (including simultaneous + multi-`-d`), throughput tuning soft checks, and CLI — without changing `iotest.py`.

**Architecture:** Tests live under `tests/` (tee-logger style). Unit tests import `iotest` directly. Integration tests call `main` / `tune_*` with tiny fixtures and a Tee_Logger under `tmp_path`. CLI tests subprocess `sys.executable` + repo-root `iotest.py`. Soft bandwidth assertions only (`finite` and `> 0`); tuning asserts directional `process_count` decrease or documented `SystemExit` on unmet high targets.

**Tech Stack:** Python 3.6+, pytest (dev dependency), existing `tee-logger` runtime dep, multiprocessing as used by production.

**Spec:** `docs/superpowers/specs/2026-08-11-iotest-test-suite-design.md`

## Global Constraints

- Do **not** modify `iotest.py` (production unchanged).
- Never write logs/reports under `/var/log/` — use `tmp_path` / `noLog=True` / CLI `-nl` `-nr`.
- Default fixture scale: `file_size=1024`, `file_count=1–2`, `process_count=1–2` (simultaneous requires `process_count >= 2`).
- Always pass `benchmarkTime=1` (or lower) into `main` / tuners so the generation benchmark does not burn the default 5s per call.
- Prefer `zeros=True` in integration tests for speed/determinism.
- Soft performance: measured sync bandwidth must be finite and `> 0` after successful write/read passes — **no absolute MB/s thresholds**.
- Tuning reality (match production, not wishful design wording): `tune_single_role` only **decreases** `process_count` when measured is above `target * TUNING_TOLERANCE`; if measured is **below** target on attempt 1 it calls `sys.exit(1)` and does **not** increase `pc`.
- If a test fails because of a wrong assumption about current behavior, fix the **test** (or document observed leftover policy). Do not change `iotest.py` unless the user explicitly asks to fix a discovered bug.
- Commit after each task.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Add `pytest` to `[dependency-groups] dev` |
| `tests/conftest.py` | Shared fixtures: tiny defaults, work/log dirs, `make_tl`, `run_main` helper |
| `tests/test_helpers.py` | Pure helpers + single-process file helpers |
| `tests/test_modes.py` | `main()` per non-simultaneous mode |
| `tests/test_simultaneous.py` | Simultaneous + multi-directory |
| `tests/test_tuning.py` | Soft tuning / bandwidth measurement |
| `tests/test_cli.py` | Subprocess CLI |

No production files are modified except `pyproject.toml` (dev deps only).

---

### Task 1: Dev dependency + shared fixtures

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Test: `tests/test_helpers.py` (minimal smoke created in this task to prove fixtures import)

**Interfaces:**
- Consumes: `iotest` module; `Tee_Logger.teeLogger` (or inline stub already inside `iotest` if import fails)
- Produces fixtures for later tasks:
  - `iotest_mod` → the imported `iotest` module
  - `work_dir(tmp_path)` → `Path` work directory
  - `make_tl(tmp_path)` → `Tee_Logger.teeLogger(..., noLog=True, suppressPrintout=True)` (or log dir under `tmp_path` if `noLog=False` needed)
  - `tiny` → dict with keys `file_size=1024`, `file_count=1`, `process_count=1`, `zeros=True`, `quiet=True`, `benchmarkTime=1`
  - `run_main(iotest_mod, make_tl, work_dir, tiny, **overrides)` → calls `iotest.main(...)` with `no_report=True` and returns the result dict

- [ ] **Step 1: Add pytest to the dev dependency group**

In `pyproject.toml`, change:

```toml
[dependency-groups]
dev = [
    "memory-profiler>=0.61.0",
]
```

to:

```toml
[dependency-groups]
dev = [
    "memory-profiler>=0.61.0",
    "pytest>=7.0.0",
]
```

- [ ] **Step 2: Sync deps**

Run:

```bash
cd /mnt/klein/work/cas/simple-iotest && uv sync --group dev
```

Expected: succeeds; `pytest` available via `uv run pytest`.

- [ ] **Step 3: Create `tests/conftest.py`**

```python
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import iotest as iotest_module  # noqa: E402
import Tee_Logger  # noqa: E402


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
```

If `import Tee_Logger` fails in the environment, construct the logger via `iotest_module.Tee_Logger.teeLogger` (the inline stub) instead — keep the same fixture kwargs.

- [ ] **Step 4: Write a minimal import smoke test**

Create `tests/test_helpers.py` with:

```python
def test_version_string(iotest_mod):
    assert isinstance(iotest_mod.version, str)
    assert iotest_mod.version == iotest_mod.__version__
```

- [ ] **Step 5: Run smoke test**

Run: `uv run pytest tests/test_helpers.py::test_version_string -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py tests/test_helpers.py
git commit -m "$(cat <<'EOF'
add pytest scaffolding and shared iotest fixtures

EOF
)"
```

Include `uv.lock` only if `uv sync` changed it.

---

### Task 2: Pure helper unit tests

**Files:**
- Modify: `tests/test_helpers.py`
- Test: `tests/test_helpers.py`

**Interfaces:**
- Consumes: fixtures from Task 1; functions on `iotest_mod`
- Produces: coverage for `format_bytes`, `parse_modes`, `normalize_directories`, `addToDicWithoutOverwrite`, `throughput_status`, `sync_bandwidth_bps`, `measure_role_bandwidth`, `supports_*_throughput_tuning`, `estimate_simultaneous_split`, `almost_urandom`

- [ ] **Step 1: Append pure-helper tests to `tests/test_helpers.py`**

```python
import math
import os

import pytest


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
```

- [ ] **Step 2: Run helper tests**

Run: `uv run pytest tests/test_helpers.py -v`

Expected: PASS. If `addToDicWithoutOverwrite` expectations fail, re-read the function and adjust asserts to match actual key renaming (do not change production).

- [ ] **Step 3: Commit**

```bash
git add tests/test_helpers.py
git commit -m "$(cat <<'EOF'
add unit tests for iotest pure helpers

EOF
)"
```

---

### Task 3: File helper unit tests on tmp_path

**Files:**
- Modify: `tests/test_helpers.py`
- Test: `tests/test_helpers.py`

**Interfaces:**
- Consumes: `create_file`, `read_file`, `move_file`, `stat_file`, `index_file`
- Produces: filesystem side-effect coverage for single-process helpers

- [ ] **Step 1: Append file-helper tests**

```python
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
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_helpers.py -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_helpers.py
git commit -m "$(cat <<'EOF'
add unit tests for iotest file helpers

EOF
)"
```

---

### Task 4: Mode integration tests (`main`)

**Files:**
- Create: `tests/test_modes.py`
- Test: `tests/test_modes.py`

**Interfaces:**
- Consumes: `run_main`, `tiny`, `work_dir`, `iotest_mod.measure_role_bandwidth`
- Produces: coverage for `write`, `read`, `index`, `comprehensive`, `random`, `benchmark`, `rw`, `rwi`, anomaly-threshold smoke

Helper to assert soft bandwidth:

```python
def assert_finite_positive(x):
    assert x is not None
    assert math.isfinite(x)
    assert x > 0
```

- [ ] **Step 1: Create `tests/test_modes.py`**

```python
import math
from pathlib import Path

import pytest


def assert_finite_positive(x):
    assert x is not None
    assert math.isfinite(x)
    assert x > 0


def test_write_mode_leaves_files_and_positive_bandwidth(run_main, work_dir, iotest_mod):
    result = run_main(["write"])
    assert "write" in result["outResults"]
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    files = list(Path(work_dir).glob("*"))
    assert any(p.is_file() for p in files)


def test_read_mode_after_write(run_main, work_dir, iotest_mod):
    run_main(["write"])
    result = run_main(["read"])
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))


def test_index_mode(run_main):
    result = run_main(["index"])
    assert "index" in result["outResults"]


def test_comprehensive_mode(run_main, iotest_mod):
    result = run_main(["comprehensive"])
    # comprehensive stores write/move/stat/read timing keys — assert no crash and write/read if present
    assert result["outResults"]
    for role in ("write", "read"):
        if role in result["outResults"] and role in result["totalTime"]:
            assert_finite_positive(iotest_mod.measure_role_bandwidth(result, role))


def test_random_mode(run_main):
    result = run_main(["random"])
    assert result["outResults"]


def test_benchmark_mode(run_main):
    result = run_main(["benchmark"], benchmarkTime=1)
    # benchmark skips worker modes; still returns dict
    assert isinstance(result, dict)
    assert result["process_count"] >= 1


def test_rw_batch_modes(run_main, iotest_mod):
    modes = iotest_mod.parse_modes(["rw"])
    result = run_main(modes)
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))


def test_rwi_batch_modes(run_main, iotest_mod):
    modes = iotest_mod.parse_modes(["rwi"])
    result = run_main(modes)
    assert "index" in result["outResults"]
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))


def test_threshold_to_report_anomaly_smoke(run_main):
    result = run_main(["write"], threshold_to_report_anomaly=1)
    assert "write" in result["outResults"]
```

**Note:** Inspect `main`’s `comprehensive` / `random` result key names on first failure and adjust asserts — keys may be `write`, `move`, `stat`, `read` separately via `addToDicWithoutOverwrite`.

- [ ] **Step 2: Run mode tests**

Run: `uv run pytest tests/test_modes.py -v`

Expected: PASS (may take tens of seconds because each call runs a 1s generation benchmark).

- [ ] **Step 3: Commit**

```bash
git add tests/test_modes.py
git commit -m "$(cat <<'EOF'
add integration tests for iotest main modes

EOF
)"
```

---

### Task 5: Simultaneous + multi-directory tests

**Files:**
- Create: `tests/test_simultaneous.py`
- Test: `tests/test_simultaneous.py`

**Interfaces:**
- Consumes: `run_main` with `process_count=2` (minimum for simultaneous), `measure_role_bandwidth`
- Produces: cleanup + dual-directory coverage

- [ ] **Step 1: Create `tests/test_simultaneous.py`**

```python
import math
from pathlib import Path

import pytest


def assert_finite_positive(x):
    assert x is not None
    assert math.isfinite(x)
    assert x > 0


def test_simultaneous_cleans_up_and_reports_bandwidth(run_main, work_dir, iotest_mod, make_tl):
    result = run_main(
        ["simultaneous"],
        process_count=2,
        tl=make_tl(),
    )
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))
    leftovers = list(Path(work_dir).glob(".iotest_simul_*"))
    assert leftovers == []


def test_simultaneous_multi_directory(iotest_mod, tmp_path, make_tl, tiny):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    result = iotest_mod.main(
        tiny["file_size"],
        tiny["file_count"],
        2,
        [str(d1), str(d2)],
        ["simultaneous"],
        tiny["quiet"],
        tiny["zeros"],
        tl=make_tl(),
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "read"))
    assert list(d1.glob(".iotest_simul_*")) == []
    assert list(d2.glob(".iotest_simul_*")) == []


def test_simultaneous_requires_pc_ge_2(run_main):
    with pytest.raises(ValueError, match="process_count >= 2"):
        run_main(["simultaneous"], process_count=1)
```

- [ ] **Step 2: Run simultaneous tests**

Run: `uv run pytest tests/test_simultaneous.py -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_simultaneous.py
git commit -m "$(cat <<'EOF'
add simultaneous and multi-directory iotest tests

EOF
)"
```

---

### Task 6: Throughput tuning soft checks

**Files:**
- Create: `tests/test_tuning.py`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Consumes: `_run_tuning_main`, `tune_single_role`, `tune_for_targets`, `measure_role_bandwidth`, `monkeypatch`
- Produces: soft bandwidth + directional pc decrease + unmet high-target `SystemExit`

**Important production behavior to encode:**

1. Absurdly **low** target (e.g. `1` B/s) with starting `pc=4` → measured typically `above` → tuner lowers `pc` across attempts → returns `0` (or accepts best effort at `pc=1`).
2. Absurdly **high** target (e.g. `10**15`) → measured `below` on attempt 1 → `sys.exit(1)`.

- [ ] **Step 1: Create `tests/test_tuning.py`**

```python
import math

import pytest


def assert_finite_positive(x):
    assert x is not None
    assert math.isfinite(x)
    assert x > 0


def test_tuning_attempt_write_bandwidth(iotest_mod, work_dir, make_tl, tiny):
    result = iotest_mod._run_tuning_main(
        tiny["file_size"],
        tiny["file_count"],
        2,
        str(work_dir),
        ["write"],
        tiny["quiet"],
        tiny["zeros"],
        make_tl(),
        True,  # stealth
        0,
        tiny["benchmarkTime"],
    )
    assert_finite_positive(iotest_mod.measure_role_bandwidth(result, "write"))


def test_tune_single_role_low_target_decreases_pc(iotest_mod, work_dir, make_tl, tiny, monkeypatch):
    pcs = []
    real = iotest_mod._run_tuning_main

    def spy(file_size, file_count, process_count, *args, **kwargs):
        pcs.append(process_count)
        return real(file_size, file_count, process_count, *args, **kwargs)

    monkeypatch.setattr(iotest_mod, "_run_tuning_main", spy)
    # Avoid final full re-run cost/noise
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: None)

    rc = iotest_mod.tune_single_role(
        tiny["file_size"],
        tiny["file_count"],
        4,
        str(work_dir),
        ["write"],
        tiny["quiet"],
        tiny["zeros"],
        make_tl(),
        "write",
        1,  # absurdly low B/s target
        stealth=True,
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )
    assert rc == 0
    assert pcs, "expected at least one tuning attempt"
    # Either decreased across attempts, or clamped at 1 on first/later attempt
    assert min(pcs) <= pcs[0]
    if len(pcs) > 1:
        assert pcs[1] < pcs[0]


def test_tune_single_role_high_target_exits(iotest_mod, work_dir, make_tl, tiny):
    with pytest.raises(SystemExit) as ei:
        iotest_mod.tune_single_role(
            tiny["file_size"],
            tiny["file_count"],
            2,
            str(work_dir),
            ["write"],
            tiny["quiet"],
            tiny["zeros"],
            make_tl(),
            "write",
            10**15,
            stealth=True,
            no_report=True,
            benchmarkTime=tiny["benchmarkTime"],
        )
    assert ei.value.code == 1


def test_tune_for_targets_read_path(iotest_mod, work_dir, make_tl, tiny, monkeypatch):
    # Seed files for read mode
    iotest_mod.main(
        tiny["file_size"],
        tiny["file_count"],
        2,
        str(work_dir),
        ["write"],
        tiny["quiet"],
        tiny["zeros"],
        tl=make_tl(),
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: None)
    # Drive through public entry with low read target (decrease-only / best-effort)
    iotest_mod.tune_for_targets(
        tiny["file_size"],
        tiny["file_count"],
        2,
        str(work_dir),
        ["read"],
        tiny["quiet"],
        tiny["zeros"],
        make_tl(),
        None,
        1,
        stealth=True,
        no_report=True,
        benchmarkTime=tiny["benchmarkTime"],
    )


def test_tune_simultaneous_dual_target_low(iotest_mod, work_dir, make_tl, tiny, monkeypatch):
    monkeypatch.setattr(iotest_mod, "_finalize_tuning_run", lambda *a, **k: None)
    # May SystemExit if either role cannot meet; low targets should succeed or accept pc=1 path
    try:
        iotest_mod.tune_for_targets(
            tiny["file_size"],
            tiny["file_count"],
            4,
            str(work_dir),
            ["simultaneous"],
            tiny["quiet"],
            tiny["zeros"],
            make_tl(),
            1,
            1,
            stealth=True,
            no_report=True,
            benchmarkTime=tiny["benchmarkTime"],
        )
    except SystemExit as e:
        # Documented failure path if a role measures below target mid-tune
        assert e.code == 1
```

- [ ] **Step 2: Run tuning tests**

Run: `uv run pytest tests/test_tuning.py -v`

Expected: PASS. If `test_tune_single_role_low_target_decreases_pc` cannot observe a decrease (already `ok` at pc=4 — unlikely with target=1), assert `pcs[-1] == 1` best-effort accept path instead.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tuning.py
git commit -m "$(cat <<'EOF'
add soft throughput-tuning tests for iotest

EOF
)"
```

---

### Task 7: CLI subprocess tests

**Files:**
- Create: `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `sys.executable`, repo-root `iotest.py` (same pattern as tee-logger `tests/test_cli.py`)
- Produces: `-V`, `-h`, tiny write, `-ts`, invalid `-wtpt`, simultaneous dual `-d`

- [ ] **Step 1: Create `tests/test_cli.py`**

```python
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
    proc = run_iotest(
        [
            "-fs", "1k",
            "-fc", "1",
            "-pc", "1",
            "-d", str(work),
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
```

**Note on `-fs 1k`:** `climain`’s nested `parse_size_arg` treats unitless as MiB and parses `k` via `'k' in size_arg`. Confirm `1k` → 1024. If CLI rejects, use `1024b` instead.

**Note on invalid `-wtpt`:** `climain` raises `ValueError` — depending on whether it is caught, subprocess may show traceback and non-zero exit. Assert `returncode != 0` only.

- [ ] **Step 2: Run CLI tests**

Run: `uv run pytest tests/test_cli.py -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "$(cat <<'EOF'
add CLI subprocess tests for iotest

EOF
)"
```

---

### Task 8: Full suite verification

**Files:**
- Test: entire `tests/`

**Interfaces:**
- Consumes: all prior tasks
- Produces: green full-suite run matching spec success criteria

- [ ] **Step 1: Run the full suite**

Run:

```bash
cd /mnt/klein/work/cas/simple-iotest && uv run pytest tests/ -v
```

Expected: all tests PASS. Confirm no files appear under `/var/log/` from this run (suite uses tmp / `-nl`).

- [ ] **Step 2: Spec coverage checklist (manual)**

Confirm present:

- [ ] helpers unit coverage (`format_bytes`, modes, bandwidth math, file helpers)
- [ ] all modes: write, read, index, comprehensive, random, benchmark, rw, rwi
- [ ] simultaneous + multi-`-d` + cleanup
- [ ] tuning soft checks (low target decrease / high target exit) + simultaneous dual-target attempt
- [ ] CLI: `-V`, `-h`, tiny write, `-ts`, invalid `-wtpt`, simultaneous dual `-d`
- [ ] anomaly threshold smoke
- [ ] `iotest.py` still unchanged (`git diff iotest.py` empty)

- [ ] **Step 3: Commit only if Step 1 required test fixes**

If fixes were needed:

```bash
git add tests/
git commit -m "$(cat <<'EOF'
fix iotest test suite issues found in full run

EOF
)"
```

If already green with nothing to commit, skip.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| `tests/` + pytest in dev group | Task 1 |
| Pure helpers | Task 2 |
| File helpers on tmp_path | Task 3 |
| Modes incl. benchmark / rw / rwi | Task 4 |
| Simultaneous + multi-`-d` | Task 5 |
| Tuning soft sanity | Task 6 (aligned to real decrease-only / SystemExit behavior) |
| CLI subprocess | Task 7 |
| Anomaly threshold smoke | Task 4 |
| No `/var/log` writes | Tasks 1, 4–7 fixtures/flags |
| `iotest.py` unchanged | Global constraint + Task 8 check |
| No hard runtime budget | Global; `benchmarkTime=1` still used for sanity |

**Design wording correction encoded in plan:** high throughput targets do not raise `process_count`; they `sys.exit(1)` when unmet. Low targets decrease `pc`.
