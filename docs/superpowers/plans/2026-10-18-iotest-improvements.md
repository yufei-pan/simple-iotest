# iotest improvements — implementation plan

Source: the 2026-10-18 review of simple-iotest 3.69.1 (commit ae1bf92), which found three bugs,
four measurement-trust problems, missing features and some code/packaging debt. The user chose:

- all four groups in scope: bugs + measurement fixes, output features, I/O features, cleanup;
- report filenames: keep today's names, shorten only when a name would be too long;
- the per-operation timeline goes to a new, separate file (the existing CSV is unchanged);
- git: work on branch `iotest-improvements`, stop before merging; no version bump, no PyPI.

No separate spec exists; this plan is the authority, and rulings made during execution are
recorded in the SDD ledger.

## Global Constraints

1. All code stays in the single module `iotest.py` (copy-transportable, importable standalone).
   No new runtime dependencies; optional ones (numpy) are imported defensively, as today.
2. `iotest.py` and the tests stay Python 3.6 compatible: no dataclasses, no walrus, no f-string
   `=`, no positional-only parameters, no `from __future__ import annotations`, no
   `subprocess.run(capture_output=..., text=...)` (use `stdout=subprocess.PIPE,
   universal_newlines=True`). Check with
   `uvx vermin -t=3.6- --no-tips --violations iotest.py tests` (must say minimum 3.6).
3. `iotest.py` is indented with tabs. Match the surrounding code: snake_case for new names, sparse
   lowercase comments that explain why, no docstrings on small helpers.
4. Tests are plain pytest `test_*` functions under `tests/`, reusing the fixtures in
   `tests/conftest.py` (`run_main`, `rec_tl`, `make_tl`, `work_dir`, `tiny`, `webhook`). The full
   suite, `uv run python -m pytest -q`, must pass with no new warnings (baseline: 237 passed,
   1 skipped). Tests must not depend on real disk speed. Tests run as root here, so
   permission-based failures (chmod) do not work; force failures another way.
5. A task that changes user-visible behavior updates `README.md` in the same task.
6. Exit codes: 0 success; 1 failed I/O ops, not enough free space, tuning failure (and, from
   Task 2, a result file that could not be written); 2 invalid arguments, always via argparse's
   `parser.error` so no traceback is printed.
7. Do not change `version`, `COMMIT_DATE` or the version in `pyproject.toml`; do not touch other
   repositories; do not push.
8. Commit on branch `iotest-improvements` with lowercase imperative subjects like the history
   ("fix ...", "add ..."), each message ending with the trailer
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
9. Text the tool prints today stays byte-identical unless the task says to change it.

## Background: how iotest.py is laid out (at ae1bf92)

- File helpers: `create_file`, `read_file`, `move_file`, `stat_file`, `index_file` return
  `(start, end)` perf_counter pairs on success and `(None, error_text)` on failure; `op_time`
  turns that into a duration or the error string.
- `worker()` runs one plain pass (write/read/random/comprehensive/index) for one process;
  `simultaneous_role_worker()` and `layout_read_worker()` serve simultaneous mode. Workers wait
  for a shared `thread_start_time` (perf_counter deadline) and hand their results to the parent
  once, at the end, through a `multiprocessing.Manager` list.
- `main()` (about 23 parameters) does the memory and free-space checks, the generation
  benchmark (only used to size the start delay), runs each pass, writes the CSV, builds and
  prints the text report, writes the report file, cleans up and sends short reports. It
  returns a dict: `outResults`, `totalTime`, `process_count`, `file_size`, `file_count`,
  `mode_process_counts`, `failedOps`.
- Result keys repeat when a mode runs twice; `addToDicWithoutOverwrite` renames them
  (`write` → `write_1`, `write_2`), and dicts that must stay aligned are updated in lockstep
  (see `add_op_results`).
- Throughput tuning (`tune_for_targets`, `tune_single_role`, `tune_dual_role_shared_pc`,
  `tune_simultaneous`, `_run_tuning_main`, `_finalize_tuning_run`) calls `main()` repeatedly,
  passing ~15 arguments positionally at every call site.
- `climain()` parses arguments and calls `main()` or `tune_for_targets()`.

## Task 1: Introduce RunOptions and tidy shared code (no behavior change)

Goal: later tasks add options in one place, and duplicated code goes away. Nothing the user
sees changes.

1. Add `class RunOptions` (a plain class, Python 3.6) holding every option of one invocation:
   `file_size`, `file_count`, `process_count`, `modes` (required), and `directories` (default
   `None`, normalized with `normalize_directories`, so a str, list or None is accepted),
   `quiet=True`, `zeros=False`, `stealth=False`, `message_end_point_address=None`,
   `no_report=False`, `threshold_to_report_anomaly=0`, `benchmark_time=5`, `unique=False`,
   `cleanup=False`. Keyword arguments only; an unknown name raises `TypeError`.
   `replace(**changes)` returns a new RunOptions and leaves the original untouched. Later tasks
   add fields here.
2. Move `main()`'s body into
   `run_iotest(options, tl=None, tuning_attempt=False, simultaneous_write_count=None,
   simultaneous_run_id=None, simultaneous_skip_layout=False, simultaneous_skip_cleanup=False,
   simultaneous_layout_workers=None, simultaneous_layout_process_count=None)`.
   Keep `main(...)` with exactly its current signature and defaults, as a thin wrapper that
   builds a RunOptions and calls `run_iotest`. Every existing test that calls `main()` passes
   unchanged.
3. The tuning functions take the options object instead of long argument lists:
   - `_run_tuning_main(options, tl, process_count, simultaneous_write_count=None, simultaneous_session=None)`
   - `_finalize_tuning_run(options, tl, process_count, simultaneous_write_count=None, simultaneous_session=None)`
   - `tune_single_role(options, tl, role, target_bps)`
   - `tune_dual_role_shared_pc(options, tl, write_target, read_target)`
   - `tune_simultaneous(options, tl, write_target, read_target)`
   - `tune_for_targets(options, tl, write_target, read_target)` and `_tune_for_targets` the same.
   Behavior is identical, including tuning attempts forcing no report and no short report, and
   `tune_simultaneous` overriding the benchmark time. Update `tests/test_tuning.py` (and any
   other test calling these) mechanically: same scenarios, same assertions.
4. `climain()` builds one RunOptions from the arguments and calls `run_iotest` or
   `tune_for_targets`.
5. Remove duplication:
   a. `run_iotest` uses `normalize_directories` instead of its own copy of that logic.
   b. `climain()`'s two echo blocks (stealth uses `tl.info`, otherwise `tl.teeprint`) become one
      block that picks the function once.
   c. The colored per-worker log lines (`'\033[38;2;{r};{g};{b}m' + ... + '\033[0m'`, about a
      dozen copies in `worker` and `simultaneous_role_worker`) go through one helper. The
      emitted text stays byte-identical.
6. Narrow the bare `except:` clauses in `create_file` and `read_file` (around posix_fadvise,
   the writev → `f.write` fallback and the readv → `f.readinto` fallback) to `except OSError`.
   Keep the fallbacks (the README documents them); KeyboardInterrupt must no longer be
   swallowed.
7. The Tee_Logger version gate compares versions as integer tuples, so `6.100` counts as newer
   than `6.34`. A version that does not parse still falls back to the inline logger.
8. Tests: RunOptions (`replace` returns a new object and leaves the original unchanged; an
   unknown option raises TypeError; a str directory becomes a one-item list, None becomes the
   cwd) and the version comparison (`'6.100'` ≥ `'6.34'`, `'6.9'` < `'6.34'`, `'7.0'` ≥ `'6.34'`,
   `'6.34'` ≥ `'6.34'`).

Done when: the full suite passes, vermin reports 3.6, and no user-visible string changed.

## Task 2: Never lose results; invalid arguments exit 2

1. One helper builds the base name of every result file:
   `iotest_{modes}_{dir_str_repr}_fs={file_size}_fc={file_count}_pc={process_count}_{datetime}`
   (exactly today's pattern). The suffixes live in one module-level tuple,
   `REPORT_SUFFIXES = ('.csv', '_report.txt')`; later tasks append theirs. When the base plus the
   longest suffix would exceed 255 bytes (UTF-8), shorten only the directory part: keep its
   longest prefix that fits, then `~` and the first 8 hex digits of the SHA-1 of the full
   directory part. Names that fit today stay exactly the same.
2. Print the text report before writing any result file.
3. Every result-file write is guarded: on OSError, `tl.teeerror(f"Could not write {path}: {e}")`
   and continue with the next file. The returned dict gains `'report_errors'`: a list of those
   messages (empty when all went well). `climain()` exits 1 when it is non-empty, after the
   leftover hint; `_finalize_tuning_run` exits 1 for it too. Short reports are still sent.
4. Fix `dir_str_repr` being defined only inside the CSV branch.
5. `-fs`, `-ts`, `-wtpt` and `-rtpt` are parsed by argparse `type=` functions built on
   `parse_size_arg` (unitless still means MiB; `-fs` still defaults to 30 MiB). An unparseable
   value exits 2 with argparse's usual `argument -fs/--file_size: ...` message naming the bad
   value. `-fs` below 1 byte and `-ts`, `-wtpt`, `-rtpt` at or below 0 exit 2. `-wtpt` without
   a write pass and `-rtpt` without a read pass go through `parser.error` (exit 2) with today's
   wording.
6. README: the exit-code section lists "a result file could not be written" under 1.
7. Tests: several long `-d` directories (total well over 255 bytes) → the run succeeds, the
   report is printed, the result files exist with names ≤ 255 bytes containing `~` plus 8 hex
   digits; a short directory keeps today's exact name; a failing result-file write (forced
   without chmod, e.g. by monkeypatching) → report still printed, `report_errors` non-empty,
   CLI exit 1; every invalid argument above → exit 2 and no `Traceback` in the output.

## Task 3: Measurement fixes

1. `create_file`: remove the posix_fadvise call before writing (it acts on a just-truncated,
   empty file). After `os.fsync`, and after `end_write_time` is taken, call
   `posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED)` where available so written data does
   not stay in page cache. Failures stay quiet as today.
2. macOS: where `fcntl.F_NOCACHE` exists, set it (`fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)`) on
   every test file opened for writing or reading, before timing starts. Import fcntl
   defensively (Windows has none).
3. Late starts: plain workers and simultaneous role workers report how many seconds late they
   reached the start barrier (only when > 0) to the parent through a Manager list passed in.
   For each pass with late workers:
   - each of that pass's report sections gets the line
     `Late starts:           \t{n} of {p} workers, up to {max:.4f} s late (sync figures include the delay)`
     (label padded like the neighbouring lines);
   - one `tl.teelog(..., level='warning')` per pass, shown without `-v`.
   The returned dict gains `'late_starts'`: `{key: (n, max_seconds)}` kept in lockstep with
   `outResults` keys (renamed together, like `failedOps`). The existing per-worker verbose
   message stays.
4. `random` mode reports reads and writes separately, under keys `random_write` and
   `random_read` (only kinds that occurred). Both use the pass's wall time as their total time
   and the pass's process count, so both get bandwidth lines. Failures from a worker that died
   before reporting (`dead_worker_errors`) go under key `random`, since their kind is unknown.
   CSV columns follow the keys.
5. comprehensive mode: each report section of a comprehensive pass ends (before the dashed
   separator) with the line
   `Note: sync figures are estimates: comprehensive mode splits the pass time across write/move/stat/read by their share of op time.`
   Track estimated keys in a dict kept in lockstep with `outResults`; the returned dict gains
   `'sync_estimated'`: `{key: bool}`.
6. README: the late-start line, `random_write`/`random_read`, the comprehensive note, macOS
   F_NOCACHE, and that on Windows reads may be served from cache.
7. Tests: fadvise runs after fsync and not before the write (record call order with
   monkeypatch); F_NOCACHE is set when the constant exists (fake it with monkeypatch); a late
   worker (e.g. call `worker()` in-process with a start time in the past) is recorded and the
   report shows the line; random mode yields `random_write`/`random_read` with bandwidth lines;
   comprehensive sections carry the note and `sync_estimated` is True; a plain write pass has
   no note.

## Task 4: Shared summary statistics and --json

1. Move the per-key statistics out of the report loop into one function that returns plain
   dicts of raw numbers; the text report and the JSON are both built from them. Text output
   stays byte-identical.
2. `-j/--json PATH` writes a JSON summary to PATH after the text report; `-` prints it to
   stdout instead (with `-S`, stdout then carries only the JSON). It is written even with
   `-nr`. A failed write is handled like other result files (`report_errors`, exit 1).
3. JSON structure, keys exactly:
   ```
   {
     "tool": "iotest",
     "version": "<version>",
     "host": "<socket.gethostname()>",
     "started": "<local ISO-8601 with UTC offset, seconds precision>",
     "options": {"file_size", "file_count", "process_count", "directories", "modes", "zeros", "unique"},
     "results": {
       "<key>": {
         "ops": <successful ops>, "failed": <failed ops>, "first_error": <str or null>,
         "process_count": <effective process count>,
         "total_bytes": <int, or null for keys without read/write>,
         "bytes_per_s_call": <float or null>, "bytes_per_s_sync": <float or null>,
         "io_time_pct": <float or null>,
         "iops_call": <float or null>, "iops_sync": <float or null>,
         "latency_s": {"avg", "median", "p1", "p99", "p99_9", "max"} or null when no op succeeded,
         "sync_estimated": <bool>,
         "late_workers": <int>, "max_late_s": <float>
       }
     },
     "failed_ops": <total failed ops>,
     "report_errors": [<str>, ...]
   }
   ```
   `options.process_count` is the requested count; `results.<key>.process_count` is the one
   that ran (the memory check can lower it). Later tasks add their options to `options`.
4. Tuning: the JSON describes the final run; `_finalize_tuning_run` runs it when `--json` is
   given, even with `-nr` and no `-addr`.
5. RunOptions gains `json_path=None`; `main()` gains a trailing `json_path=None` keyword.
6. README: document `--json` and the structure briefly.
7. Tests: JSON keys and types for a write+read run; `-j - -S` via the CLI → stdout parses as
   JSON; an unwritable JSON path → exit 1; the text report formatting is unchanged (existing
   tests plus one test of the formatter on fixed statistics).

## Task 5: Per-operation timeline CSV

1. Whenever result files are written (not `-nr`, not a tuning attempt), also write
   `<base>_ops.csv` (append `'_ops.csv'` to `REPORT_SUFFIXES`), using the csv module, with
   header `pass,mode,op,worker,file_index,start_s,duration_s,error` and one row per attempted
   op:
   - `pass`: 1-based index of the pass in this run (the generation benchmark does not count;
     a simultaneous run is one pass);
   - `mode`: the pass's mode (`write`, `read`, `random`, `comprehensive`, `index`,
     `simultaneous`);
   - `op`: `write`, `read`, `move`, `stat` or `index`;
   - `worker`: the worker's counter name as used for its directory;
   - `file_index`: 1-based;
   - `start_s`: seconds from the pass's start barrier to the op's start, 6 decimals; for a
     failed op, when the attempt began;
   - `duration_s`: 6 decimals, empty for a failed op;
   - `error`: empty on success, the error text on failure.
   Rows are sorted by pass, then `start_s`. Workers that died before reporting have no rows.
2. Workers collect rows only when the timeline will be written (pass `None` otherwise), so
   tuning attempts stay cheap.
3. README: describe the file and its columns.
4. Tests: a write+read run → `pc*fc` rows per pass, pass 1 is write and pass 2 is read,
   `start_s` ≥ 0, durations equal the main CSV's values as multisets; a failed op has an error
   and an empty duration; `-nr` → no `_ops.csv`.

## Task 6: --start_at for synchronized multi-host starts

1. `--start_at EPOCH`: seconds since the Unix epoch (float), e.g. `$(date -d '+1 min' +%s)`.
   Not a number, or not in the future when arguments are parsed → exit 2. Combined with
   `-wtpt`/`-rtpt` → exit 2 (`--start_at cannot be combined with throughput tuning`).
2. The first timed pass (the first mode other than benchmark; for simultaneous, after its
   layout pass) uses the start barrier `time.perf_counter() + (start_at - time.time())`,
   computed when that pass is scheduled. If less than the usual process start delay remains,
   it starts at `perf_counter() + processStartDelay` instead, with a `tl.teelog` warning saying
   how many seconds late it starts. Later passes start as they do today.
3. When `--start_at` is used, the text report starts with one line:
   `Synchronized start at <local ISO-8601 time> (--start_at)`; if the start was late, a second
   line says by how much.
4. RunOptions gains `start_at=None`; `main()` gains a trailing `start_at=None`; the JSON
   `options` include `start_at`.
5. README: a "Multi-host runs" section. Clocks must be in sync (NTP/chrony); only the first pass
   is synchronized. Example, computing the time once locally:
   ```bash
   T=$(date -d '+2 min' +%s)
   mssh node[1-8] "iotest -d /scratch/\$(hostname) --start_at $T -j /scratch/iotest-\$(hostname).json"
   ```
6. Tests: a start_at ~2 s ahead → the first pass's barrier is not before start_at (observe it,
   e.g. via the timeline or by monkeypatching); a past or non-numeric value → exit 2; with
   `-wtpt` → exit 2; a too-soon start_at → the warning.

## Task 7: -bs/--block_size

1. `-bs/--block_size SIZE` (same size syntax as `-fs`; unitless means MiB). Default: unset,
   meaning one write/read call per file as today. A block size ≥ the file size behaves exactly
   like unset. Below 1 byte → exit 2.
2. When set, workers hold one buffer of the block size instead of the file size. A file is
   written as consecutive block-size write calls (the last one shorter; short writes still
   retried), then fsync; it is read with block-size read calls into a block-size buffer.
   Applies to every path that writes or reads test files: write, read (including recreating
   missing files), random, comprehensive, simultaneous writers, readers and the layout pass.
   Per-op timing still covers the whole file.
3. With `-u`, every 512-byte block of the run stays unique: `stamp_unique` gains a
   `first_block=0` parameter, and each chunk is stamped with its position in the file before
   it is written (inside the timed window when chunking; say so in the README).
4. The memory estimate and the generation benchmark use the buffer size actually held
   (`min(block_size, file_size)`). The free-space check is unchanged.
5. The run header (the lines `climain()` prints: file size, number of files, ...) shows
   `Block size: <size>` when set. RunOptions gains `block_size=None`; `main()` gains a trailing
   `block_size=None`; the JSON `options` include `block_size` (null when unset).
6. README: the option; without `-u`, a chunked file repeats its block, so deduplicating storage
   may store it once (use `-u`).
7. Tests: `-bs` smaller than the file → correct file size, and with `-u` all 512-byte blocks
   unique (reuse the approach in `tests/test_unique_content.py`); reads with `-bs` succeed and
   read the whole file (short-read handling covered like `tests/test_short_io.py`); the memory
   estimate uses the block size; `-bs` ≥ file size behaves as unset; invalid `-bs` → exit 2.

## Task 8: --verify

1. `--verify` makes every test file self-checking. It implies `-u`. It needs a file size of at
   least 8 bytes, otherwise exit 2.
2. Every file written (write, random, comprehensive, simultaneous writers and layout, files
   recreated by read) ends with a 4-byte little-endian CRC-32 (`zlib.crc32`, masked to 32 bits)
   of all preceding bytes of that file. With a whole-file buffer, the CRC is computed after
   stamping and before the timed write. With `-bs`, it is accumulated across chunks and the last
   chunk carries it (inside the timed window).
3. Every read (read, random reads, comprehensive read, simultaneous readers) checks the
   trailer: for whole-file reads after the timed read call (outside the op time, inside the
   sync time); with `-bs`, accumulated while reading. A mismatch makes that op fail with the
   error `verify failed: <path>: CRC-32 mismatch (stored 0x%08x, computed 0x%08x)`, so it counts
   in the failed ops and the exit code.
4. Files written without `--verify` fail verification; the README says to write them with
   `--verify` as well.
5. RunOptions gains `verify=False`; `main()` gains a trailing `verify=False`; the JSON `options`
   include `verify`.
6. README: the option, what it detects (bit flips, torn or misplaced writes) and its CPU cost.
7. Tests: write+read with `--verify` passes; flipping one byte of a written file makes a later
   `--verify` read of that file fail with the message, and the CLI exits 1; `--verify` with
   `-bs` passes and catches a flipped byte; `-fs 4 --verify` → exit 2; `--verify` alone stamps
   blocks as `-u` does.

## Task 9: Metadata mode

1. New mode `meta` (alias `m`; add to argparse choices, `MODE_ALIASES` and the help text). It
   runs three passes in order, create → stat → unlink, each like any other pass: all workers,
   their own start barrier, wall time, late-start tracking and timeline rows (Task 5:
   `mode` `meta`, `op` `create`/`stat`/`unlink`). Results go under keys `meta_create`,
   `meta_stat` and `meta_unlink`.
2. Each worker uses `<directory>/<counter>/meta/` with files `meta_<i>` (zero-padded to the
   width of file_count). Before the create pass's barrier, each worker removes leftovers in its
   meta directory (untimed). create = `os.open(path, O_CREAT | O_EXCL | O_WRONLY, 0o644)` then
   `os.close`; stat = `os.stat`; unlink = `os.unlink`. After the unlink pass the worker removes
   its empty meta directory (untimed). A missing file in stat or unlink is a failed op.
3. Meta passes hold no data buffers, need no free-space check, and a run of only meta (and
   index/read without recreation) skips the generation benchmark. `-fs` does not apply (help
   text and README say so).
4. `-cl` also removes meta directories left behind.
5. README: the mode and its keys.
6. Tests: a meta run → three keys with `pc*fc` ops each, IOPS lines and no bandwidth lines,
   no files left; leftover meta files from an interrupted run do not cause failures; a stat of
   a missing file counts as a failed op.

## Task 10: Packaging, docs and CI

1. `pyproject.toml`, following `../tee-logger/pyproject.toml` (absolute:
   `/mnt/optane/work/cas/tee-logger/pyproject.toml`):
   - `[build-system]` `requires = ["setuptools>=61"]`, `build-backend = "setuptools.build_meta"`;
   - `dynamic = ["version"]` with `[tool.setuptools.dynamic] version = {attr = "iotest.version"}`
     (remove the static version), `[tool.setuptools] py-modules = ["iotest"]`;
   - `license = {text = "GPL-3.0-or-later"}` plus the GPLv3+ classifier;
   - optional extra `fast = ["numpy"]`;
   - drop the unused `memory-profiler` dev dependency;
   - refresh `uv.lock` (`uv lock`).
   `uv build` must produce a wheel with only `iotest.py` and dist-info at version 3.69.1, and an
   sdist that includes `tests/conftest.py` (add a MANIFEST.in if needed).
2. Add `LICENSE` with the GPL-3.0 text copied from `/mnt/optane/work/cas/multiCMD/LICENSE`.
3. README refresh: remove the stray "Generated:" line, fix typos ("Recommand", "seperate");
   `-q` is the default and `-v` gives per-file output; Tee_Logger is a required dependency,
   installed automatically, logging to /var/log by default; document every option (including
   `-v`, `-S`, `-ts`, `-ld`, `-addr` (posts Discord-style JSON `{"content": ...}`),
   `--threshold_to_report_anomaly`, `-bt`) and every mode (including `random`, `benchmark`,
   `meta`); mention `pipx install 'simple-iotest[fast]'` for numpy; regenerate the `-h` dump
   from the final `python iotest.py -h`.
4. CI (`.github/workflows/test.yml`; keep actions pinned by SHA like the existing steps):
   - a `lint` job running a pinned ruff with `[tool.ruff] target-version = "py37"` and
     `[tool.ruff.lint] select = ["E4", "E7", "E9", "F"]` in pyproject; fix what it reports;
   - `macos-latest` and `windows-latest` jobs on Python 3.12. Read the tests and mark the ones
     that need Linux (/proc, mountinfo, posix_fadvise, fork-only behavior, ...) with
     `pytest.mark.skipif(sys.platform != 'linux', reason=...)`. These jobs cannot be run here;
     the report must say they are unverified.
5. Verify: full suite, `uv build` contents, ruff, vermin.
