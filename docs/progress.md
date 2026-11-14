# simple-iotest: progress

Open work lives here and nowhere else (RULES, "Long work: the progress file"). Edit rows in place, in the commit that changes the work.

## Goal
- 2026-11-05, for every cas tool: "we improve the cas scripts iteratively. feel free to add scope / tasks if when doing some work, you noticed some other thing that can be improved with cas scripts." (the user; no goal for this repo alone is written in the repo)

## Open work

### Release riders (no written plan; the user's pick is quoted in the row)
- **Add Python 3.11 to the GitHub CI matrix.** State: not started. Where: `.github/workflows/test.yml`, the first job's matrix `["3.9", "3.12", "3.14"]` on master ab2623b; add `"3.11"` in the commit of the repo's next release push. No next release is queued (the 3.70.1 release notes say "Nothing queued"), so this rides whichever release comes next. The other jobs (numpy on 3.12, the Python 3.6 container, lint, macOS and Windows on 3.12) stay as they are. From: the user, 2026-11-12: "Add 3.11 to CI (Recommended)", for every repo's next release.

### Standing rows
- **Deferred work** is in `docs/limitations.md` (entries of kind "deferred"); each names when it is revisited. This repo has no `docs/limitations.md` yet, so nothing is deferred.

## Questions waiting for the user
- None.

## Closed
- None.
