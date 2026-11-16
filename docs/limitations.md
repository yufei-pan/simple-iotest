# Limitations

Approved known limitations and deferrals. Reviewers: read this first, and don't raise these again.

### `-pc 1` with simultaneous mode ends in a traceback
- Kind: deferred
- What the operator sees: `iotest -pc 1 simultaneous` (or `-t 1 s`, a `-ts` that works out to one process, or a run the memory check cuts to one process) prints the run settings and the process start delay, then a Python traceback ending in `ValueError: simultaneous mode requires process_count >= 2`, and exits 1. No result file is written. Run alone, simultaneous leaves the `-d` directory empty; passes before it in the mode list (as in `-pc 1 w s`) run but are not reported, and leave their test files even with `-cl`. The run log (in `-ld`, off with `-nl`) holds the settings lines, not the traceback.
- Why: found in the 3.71.0 simplify pass, which keeps behaviour unchanged; 3.70.1 does the same, and the fix is its own change.
- Revisit when: the next version. Reject it after `-ts` is applied, as an invalid argument (exit 2, "simultaneous mode needs at least 2 processes"), and let the memory check keep two processes for a simultaneous run.
