# Limitations

Approved known limitations and deferrals. Reviewers: read this first, and don't raise these again.

### `-pc 1` with simultaneous mode ends in a traceback
- Kind: deferred
- What the operator sees: `iotest -pc 1 simultaneous` (or `-t 1 s`, a `-ts` that works out to one process, or a run the memory check cuts to one process) prints the run settings and the process start delay, then a Python traceback ending in `ValueError: simultaneous mode requires process_count >= 2`, and exits 1. No result file is written. Run alone, simultaneous leaves the `-d` directory empty; passes before it in the mode list (as in `-pc 1 w s`) run but are not reported, and leave their test files even with `-cl`. The run log (in `-ld`, off with `-nl`) holds the settings lines, not the traceback.
- Why: found in the 3.71.0 simplify pass, which keeps behaviour unchanged; 3.70.1 does the same, and the fix is its own change.
- Revisit when: the next version. Reject it after `-ts` is applied, as an invalid argument (exit 2, "simultaneous mode needs at least 2 processes"), and let the memory check keep two processes for a simultaneous run.

### On macOS with Python 3.8 or 3.9, each pass waits as long as the run has lasted, and the sync bandwidth is wrong
- Kind: deferred
- What the operator sees: on macOS with Python 3.8 or 3.9 (Apple's `/usr/bin/python3` is 3.9), every pass waits before it starts for about as long as the run has already lasted, so each pass takes about twice as long as the one before, and the sync bandwidth and IOPS include that wait: `iotest -fs 1m -fc 2 -pc 2 -z -bt 2 w r w r` took 25 s and reported 6.46 MiB/s down to 339 KiB/s, where Python 3.11 on the same Mac took 4.4 s at 114 to 119 MiB/s. The per-call figures are right. A tuning run effectively never finishes. By CPython's changelog, Windows before Python 3.10 likewise (not run). Python 3.10 and later, and Linux, are not affected. 3.70.1 does the same.
- Why: there, `time.perf_counter()` counts from each process's own start, and the parent hands its spawned workers a start time on its own clock (`thread_start_time`, iotest.py:1763 and :1803; `first_pass_start`, :1536), which each worker waits for on its own (`wait_for_barrier`, :594). Found by the week-48 cas-toolbox roll-up gate on the test Mac (2026-11-21).
- Revisit when: 3.72.0. Hand the start time over on a clock all processes share (`time.clock_gettime(time.CLOCK_MONOTONIC)`, as hpcp's `_shared_clock()`), which each worker turns into its own `perf_counter()` once; keep op timing on `perf_counter()`. A test on Linux with the spawn start method and the parent's `perf_counter()` shifted by an hour.
