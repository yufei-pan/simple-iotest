# Limitations

Approved known limitations and deferrals. Reviewers: read this first, and don't raise these again.

### `-pc 1` with simultaneous mode ends in a traceback
- Kind: deferred
- What the operator sees: `iotest -pc 1 simultaneous` (or `-t 1 s`, a `-ts` that works out to one process, or a run the memory check cuts to one process) prints the run settings and the process start delay, then a Python traceback ending in `ValueError: simultaneous mode requires process_count >= 2`, and exits 1. Nothing is written.
- Why: found in the 3.71.0 simplify pass, which keeps behaviour unchanged; the fix is its own change.
- Revisit when: the next version. Reject it after `-ts` is applied, as an invalid argument (exit 2, "simultaneous mode needs at least 2 processes"), and let the memory check keep two processes for a simultaneous run.
