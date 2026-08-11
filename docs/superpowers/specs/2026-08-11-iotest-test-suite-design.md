# Design: Full test suite for simple-iotest

Date: 2026-08-11  
Status: pending user review of written spec  
Package: `simple-iotest` (`iotest.py` v3.66.0)

## Goal

Add a full pytest-based test suite covering pure helpers, all operational modes (including simultaneous and multi-directory), throughput tuning, and CLI entrypoints — without modifying production code in `iotest.py`.

## Decisions (from brainstorming)

| Topic | Choice |
|---|---|
| Scope | Full stack (unit + thin integration + tuning/simultaneous/multi-`-d`/anomaly-threshold plumbing) |
| Runtime budget | No hard wall-clock budget; correctness over speed |
| Layout / runner | `tests/` + pytest; add `pytest` to `[dependency-groups] dev` |
| Assertions | Soft performance sanity (finite bandwidth `> 0`; tuning moves `process_count` in expected direction or hits documented clamp) |
| Approach | Layered pytest against the public API; **no production changes** |

## Non-goals

- Mocking `multiprocessing` or faking worker I/O
- Absolute MB/s thresholds or cross-machine performance SLOs
- Asserting pretty-printed report / log prose
- Load/stress sizes (fixtures stay tiny)
- Windows-specific path coverage beyond what falls out naturally
- Extracting nested helpers (e.g. `parse_size_arg`) from `climain` for unit tests

## Architecture

```
tests/
  conftest.py              # shared fixtures
  test_helpers.py          # pure / file-helper units
  test_modes.py            # main() per mode
  test_simultaneous.py     # simultaneous + multi-directory
  test_tuning.py           # -wtpt / -rtpt soft sanity
  test_cli.py              # subprocess climain
```

- Tests import `iotest` as a module for helpers, `main`, and tuners.
- CLI coverage invokes `sys.executable` + repo-root `iotest.py` via `subprocess` (avoids requiring an editable install for the suite to run).
- Work and log directories always live under pytest `tmp_path` (or CLI `-nl` with log dir under tmp) so nothing writes to `/var/log/`.
- Default fixture scale: ~1 KiB file size, `file_count` 1–2, `process_count` 1–2. Prefer `pc ≤ 2` to limit process storms.
- `main(...)` returns a dict with keys `outResults`, `totalTime`, `process_count`, `file_size`, `file_count`, `mode_process_counts` — integration assertions use this return value.

### Packaging

- Add `pytest` to `[dependency-groups] dev` in `pyproject.toml`.
- Run with: `uv sync --group dev` then `uv run pytest`.
- No change to runtime dependencies of the published package.

## Coverage map

### Unit — `test_helpers.py`

| Surface | Assertions |
|---|---|
| `format_bytes` | Parse/format round-trips; SI vs 1024 units; invalid input returns `0` / passthrough as today |
| `parse_modes` | Aliases expand (`w`, `rw`/`wr`, `rwi`/`wri`, `c`, `s`, `i`, `random`, `benchmark`) |
| `normalize_directories` | Single path, list, defaults |
| `addToDicWithoutOverwrite` | First write wins |
| `throughput_status` | over / under / ok classification vs target |
| `sync_bandwidth_bps` / `measure_role_bandwidth` | Synthetic result dicts → expected bps |
| `supports_write_throughput_tuning` / `supports_read_throughput_tuning` | Mode gating |
| `estimate_simultaneous_split` | Split respects total `pc` and target ratio |
| `almost_urandom` | Returned buffer length equals request |
| `create_file` / `read_file` / `move_file` / `stat_file` / `index_file` | Correct filesystem side effects on `tmp_path`; zeros vs random content path |

### Mode integration — `test_modes.py`

Call `main(...)` with `quiet=True`, `no_report=True`, Tee_Logger rooted under `tmp_path`:

- Modes: `write`, `read` (after a write pass that leaves files), `index`, `comprehensive`, `random`, `benchmark` (with a short `benchmarkTime`, e.g. 1s), and batch expansions `rw` / `rwi` (via `parse_modes` then `main`, matching climain’s multi-mode list).
- Assert: no uncaught exception; where a sync write/read pass runs, measured bandwidth is finite and `> 0`.
- Leftover policy: write leaves files; index cleans its temporary index folders; comprehensive/random leave state consistent with current `main` behavior (document observed cleanup in the test docstring if non-obvious).

### Simultaneous / multi-dir — `test_simultaneous.py`

- End-to-end `simultaneous`: layout → concurrent write+read roles → cleanup; assert simultaneous tree is removed.
- Same flow with two directories (round-robin workers across repeated `-d` / directory list).
- Soft check: write and read role bandwidths finite and `> 0`.

### Tuning — `test_tuning.py`

- Drive `tune_for_targets` / role-specific tuners with tiny I/O and:
  - absurdly **low** target → expect `process_count` to decrease or hit the documented minimum / early stop;
  - absurdly **high** target → expect increase or hit max attempts / starting ceiling.
- Measured bps after a successful attempt: finite and `> 0`.
- If a host cannot distinguish direction in available attempts, assert the documented clamp / “gave up” path rather than failing flakily.
- Cover write-only, read-capable, and simultaneous dual-target paths at least once.

### CLI — `test_cli.py`

Subprocess cases:

- `-V` / `--version` exits 0 and prints version.
- `-h` exits 0.
- Default write with tiny `-fs`/`-fc`/`-pc`, `-d` under tmp, `-nl` `-nr` completes successfully.
- `-ts` / `--total_size` auto-fills the missing one of `-fs` / `-fc` / `-pc`.
- Invalid combos: e.g. `-wtpt` with index-only mode → non-zero exit / error.
- Simultaneous + dual `-d` smoke via CLI.

### Anomaly threshold

- At least one integration or CLI path passes a non-zero `--threshold_to_report_anomaly` with tiny fixtures to ensure the argument is accepted and does not crash (no requirement to force an anomaly condition unless cheap to synthesize).

## Error handling & isolation policy

- **Logging isolation:** Tee_Logger under `tmp_path`, or CLI `-nl` / log directory under tmp — never `/var/log/`.
- **Filesystem isolation:** unique `tmp_path` per test; simultaneous tests verify cleanup.
- **Process isolation:** keep `process_count` small (`≤ 2` unless a case specifically needs more for split math).
- **Asserted failures:** invalid `-ts` (≤0), `-wtpt`/`-rtpt` on unsupported modes → `ValueError` from `climain` (CLI non-zero exit). Do not require a dedicated “read with no prior files” failure case unless observed behavior is stable and cheap to assert.
- **Flake policy:** no absolute throughput numbers; prefer directional / clamp assertions for tuning.

## Data flow (typical integration test)

1. Fixture creates work dir + optional log dir under `tmp_path`.
2. Construct or obtain a Tee_Logger pointed at the log dir (or `noLog=True`).
3. Call `main` / `tune_for_targets` with tiny sizes and explicit modes.
4. Inspect return dict (bandwidths, `process_count`) and filesystem leftovers.
5. Fixture teardown removes `tmp_path` (pytest default).

CLI tests replace steps 2–4 with `subprocess.run([...], check=False)` and assert `returncode` / stdout / stderr.

## Testing approach summary

| Layer | Mechanism | Speed | Confidence |
|---|---|---|---|
| Helpers | Direct function calls | Fast | High for pure logic |
| Modes / simultaneous / tuning | Real multiprocessing + disk | Slower | High for real I/O paths |
| CLI | Subprocess | Slower | High for argparse / nested `parse_size_arg` / `-ts` |

## Success criteria

- `uv run pytest` passes on a POSIX host with the package’s declared dependencies available.
- All modes listed above have at least one passing integration or CLI case.
- Tuning directional/clamp soft checks are present for write and read (and simultaneous dual-target once).
- No writes outside pytest temp directories during the suite.
- `iotest.py` unchanged.

## Follow-up

After this spec is reviewed and approved as written, create an implementation plan via the writing-plans skill (task breakdown, order, and verification commands).
