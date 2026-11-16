# simple-iotest: progress

Open work lives here and nowhere else (RULES, "Long work: the progress file"). Edit rows in place, in the commit that changes the work.

## Goal
- 2026-11-05, for every cas tool: "we improve the cas scripts iteratively. feel free to add scope / tasks if when doing some work, you noticed some other thing that can be improved with cas scripts." (the user; no goal for this repo alone is written in the repo)

## Open work

### Release riders (no written plan; the user's pick is quoted in the row)
- **Built; under final review (2026-11-15):** branch feat/simplify (worktree iotest-simplify) at b0cdb81, 15 commits on a771fcf: the simplify pass (iotest.py 2,790 to 2,457 lines), Python 3.11 in CI (10a8ffd), the os.writev/os.readv retry removed (dfab87b; a chief's decision of 2026-11-15, not the user's words), version 3.71.0 (0404790), docs/limitations.md with the `-pc 1 simultaneous` deferral (b0cdb81). Suites pass on 3.14, 3.11, numpy and 3.6 (first runs); a real benchmark against 3.70.1 gives the same report lines and keys, figures in range (paused 2026-11-14 at the user's ask; resumed on chief's order after the user's "Resume both (Recommended)"). Final review (2026-11-16): release once the docs are corrected; nothing worse than 3.70.1 (27 behaviour cases, speed within 1.4%, the tuner identical on 20,000 scenarios). Fix round 1 (docs and tests only, running): README's -v, memory and -bs sentences, the deferred entry's wording and its README line, four tests. Next: a re-review of the round, then the release (COMMIT_DATE at the release commit). Deferred to the next version: tests for stat_file's failure tuple, comprehensive's wall-time split and the -v verb (also untested in 3.70.1).
- **Add Python 3.11 to the GitHub CI matrix.** State: built (10a8ffd), rides 3.71.0. Where: `.github/workflows/test.yml`, the first job's matrix `["3.9", "3.12", "3.14"]` on master ab2623b; add `"3.11"` in the commit of the repo's next release push. No next release is queued (the 3.70.1 release notes say "Nothing queued"), so this rides whichever release comes next. The other jobs (numpy on 3.12, the Python 3.6 container, lint, macOS and Windows on 3.12) stay as they are. From: the user, 2026-11-12: "Add 3.11 to CI (Recommended)", for every repo's next release.

### The simplify pass (no written plan)
- **Simplify and drop dead weight, as the other cas tools had.** State: building on feat/simplify (row above). Where: remove what no operator needs (dead code, guards for states that cannot happen, extra layers), behaviour unchanged, reviewed like a release. From: the user, 2026-10-23: "maybe we should go through the repos after this round, then simplify and drop any stuff that are dead weight?"; for firewatcher, simple-iotest and tsvz-rs, "Run it on all three (Recommended)", 2026-11-14.

### Standing rows
- **Deferred work** is in `docs/limitations.md` (entries of kind "deferred"); each names when it is revisited. This repo has no `docs/limitations.md` yet, so nothing is deferred.

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
- None.
