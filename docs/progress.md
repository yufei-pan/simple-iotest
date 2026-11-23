# simple-iotest: progress

Open work lives here and nowhere else (RULES, "Long work: the progress file"). Edit rows in place, in the commit that changes the work.

## Goal
- 2026-11-05, for every cas tool: "we improve the cas scripts iteratively. feel free to add scope / tasks if when doing some work, you noticed some other thing that can be improved with cas scripts." (the user; no goal for this repo alone is written in the repo)

## Open work

### simple-iotest 3.73.0
- **Nothing queued.** The next version starts from what 3.72.0's review or an operator finds. State: none open.

### Standing rows
- **Deferred work** is in `docs/limitations.md` (entries of kind "deferred"); each names when it is revisited. None today: 3.72.0 settled both.

## Standing decisions
The user's rulings on how the cas repos are worked, one line each; a line changes only on the user's word. The cas monorepo's CLAUDE.md holds the same list, with how to reach the test machines.
- Deferrals (2026-11-05): "please do not ask me about features you think should defer to next version anymore. instead, note them down and after getting the current version done, start the next version right away. only ask me to approve things that are going into limitations that we do not have plans to fix it any time soon."
- The release gate (2026-11-09): "releasing a new version gate is only if it does not make an existing issue worse. even if the fix does not completely resolve the problem, if it makes the result more tolerable, we shall not gate it for release because there are still issues."
- Review before upload (2026-10-18): "commit and push is fine but when packaging, please follow the rule that need a code review before uvpack / upload to pypi".
- A test that passed only on a rerun (2026-11-06, "Green with a warning (Recommended)"): CI stays green with a warning, and "Before any upload, every such test still needs a written cause either way."
- Release pushes (2026-11-10, "Release OK covers it (Recommended)"): an OK for a release covers its push of main, the version tag and the GitHub release.
- Tests (2026-11-12, "Core per release, full at roll-up (Recommended)"): CI's own command on Python 3.14 and 3.11 in Docker before each release; the full gate (every Python in CI, macOS, EL7 on Python 3.11) before a Monday roll-up pin moves.
- CentOS 7 (2026-11-04): "for centos 7, which is already EOL, we only target all the last packages released for it. this also includes EPEL which is also in archive right now."
- EL8 (2026-11-05): "yes rocky 8 it is. let's also only test on them when you think test on real kernel is needed."
- macOS (2026-11-11): tests run on the user's own always-on Mac, as a standard user made for them, never with sudo.

## Questions waiting for the user
- None. Decided 2026-11-15, a chief's decision (not the user's words): remove the os.writev/os.readv retry through f.write/f.readinto; the release notes say that a write or read now fails on its first error (built in dfab87b).

## Closed
- 3.72.0 (tag v3.72.0, the merge of feat/3720 a21e5f4, released 2026-11-23): on macOS with Python 3.8 or 3.9 passes start on time and the sync figures are right (the start is handed between processes on a clock all share, `perf_counter()` from Python 3.10, and every measurement stays on `perf_counter()`); `-pc 1` with simultaneous mode is an argument error (exit 2); the memory cut keeps 2 processes for simultaneous and the tuner goes on from the split the run used; the memory warning waits the 15 s it says; tests the 3.71.0 reviews found missing. Build b3720, final review fr3720 (Fable high: fix first: a pass's wall time counted a Mac's sleep), fix round f3720, re-review rr3720 (Opus xhigh: release after the hand-over clock from 3.10), last round f3720b; suites 605 on 3.14 and 3.11, 604 on 3.6, 603 on 3.12 and 3.9, the Mac 573 and 571, EL7 603, first runs; PyPI wheel f8d3fea3 and sdist 4c03f1e1 match the build; GitHub release v3.72.0. cas-toolbox pins it after its full gate.
- 3.71.0 (tag v3.71.0, the merge of feat/simplify, released 2026-11-16): the simplify pass (iotest.py 2,790 to 2,457 lines; the user, 2026-10-23, and "Run it on all three (Recommended)", 2026-11-14), behaviour unchanged apart from the os.writev/os.readv retry, removed (a chief's decision of 2026-11-15: a refused write or read fails on its first error); Python 3.11 in CI ("Add 3.11 to CI (Recommended)", 2026-11-12); a README trim; docs/limitations.md with the `-pc 1 simultaneous` deferral. Final review release after docs; fix round 1 docs and tests (a23910f, 5bcb224); re-review release; texts f90e6c8, 70fc1dd. Main CI green on all nine jobs; macOS green on a rerun of its job, whose first attempt never ran (GitHub: "The job was not acquired by Runner of type hosted even after multiple attempts", macOS capacity).
- None.
