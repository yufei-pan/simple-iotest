# iotest

Ever feel like fio is too complicated and too optimized? Well, simple-iotest covers this for you!

Using the most simple non-optimized write() or writev() with fallback to python f.write() (just like your code!) to test file system io performance.

Warning: if using big file sizes, make sure you have enough memory to hold all these bits! (Or use `-bs`: each worker then holds a chunk pool of up to 16 MiB and one block to write from, and a one-block read buffer, instead of whole files.) Before it starts, iotest estimates the memory the run needs: each worker's data buffers (none in index and meta passes; a read worker that finds some of its files missing also holds the content to recreate them), plus, per operation, about 80 (meta) to 140 (simultaneous) bytes for the results it keeps until the run ends, and about 70 to 110 more with `-tl`. If that is more than the available memory plus swap, it lowers the process count; if it is more than 90% of the available memory, it warns and waits 15 seconds before going on.

Note:
	Tee_Logger is a required dependency and is installed automatically. Results are also logged to `/var/log/` by default (change it with `-ld`, or turn it off with `-nl`) for future keeping.

This script tests I/O performance by creating, reading, moving, and indexing files in various modes. It allows you to measure different aspects of disk performance and log the results.

## Installation
```bash
pipx install simple-iotest
```
Optionally install numpy, which speeds up generating the random test data:
```bash
pipx install 'simple-iotest[fast]'
```

## Usage
Run the script with:
```bash
iotest [options] [modes]
simple-iotest [options] [modes]
```

`simple-iotest` is the same command as `iotest` (so `uvx simple-iotest` works).

Default mode is **write** (`w`) when no mode is specified.

### Common Options
- `-fs, --file_size`: Size of the test files (default: 30 MiB; suffix `m`, `g`, etc. supported).  
- `-bs, --block_size`: Writes and reads each file in calls of this size (same units as `-fs`), so large files need little memory and small-I/O patterns can be tested. Each worker reads into one buffer of this size, and writes from one chunk pool: 16 MiB of its test data plus one block (the whole file, if that is smaller), generated once before the timed pass. So each writer holds up to 16 MiB and one block (less for smaller files), which the memory estimate counts. Each chunk is written from its offset in the file modulo the pool's span (16 MiB and one byte; 16 MiB with `-u`), so a file is not one block repeated: a byte of the pool comes back in the file only a whole span later, never twice within any 16 MiB, ZFS's largest recordsize. No record that storage compresses on its own (ZFS records of up to 16 MiB, btrfs's 128 KiB extents, an array's blocks) contains a repeat, so compression cannot shrink the file and inflate the write figures (`-z` data still compresses). Default: unset, one call per file; a block size at least the file size is the same as unset. A short write or read is retried with the rest of its block, and a written file is synced once, after its last block. An operation is still one whole file: its time covers every block, and IOPS count files. Without `-u`, each worker's files hold the same data (as without `-bs`), so deduplicating storage may store them once; with `-u`, each chunk is stamped with its place in the file just before it is written, inside the timed write (without `-bs` the whole file is stamped before the timer starts). The tags land on the same bytes of the pool whichever chunk writes them, so every 512-byte block has its own tag and no earlier tag is left between them, but with `--verify` the previous file's CRC trailer can be, so with `-z -u --verify -bs` the bytes between tags are not all zeros.  
- `-fc, --file_count`: Number of files to process per worker (default: 50).  
- `-pc, -t, --process_count`: Number of worker processes (default: 36).  
- `-ts, --total_size`: Total size target. If one of `-fs`, `-fc` and `-pc` is omitted, it is calculated to fit this total (the file count first, then the file size, then the process count); it is ignored, with a warning, when all three are given.  
- `-wtpt, --write_target_throughput`: Target sync write throughput (e.g. `500m`, `2g`); auto-tunes the first write pass.  
- `-rtpt, --read_target_throughput`: Target sync read throughput; auto-tunes the first read pass. For simultaneous mode with both targets, the initial writer/reader split is estimated from the target ratio before tuning.  
- `-d, --directory`: Directory for file operations.  
- `-ld, --log_directory`: Directory for the log files (default: `/var/log/`).  
- `-q, --quiet`: Kept for compatibility: quiet is the default.  
- `-v, --verbose`: Prints per-file output.  
- `-S, --stealth`: Suppresses the console output and the verbose log file. It implies `-nr`, so no CSV or report files are written (`-tl` still writes the timeline). Worker progress digits still go to stdout unless `-j -` sends the JSON there (the digits then go to stderr).  
- `-z, --zeros`: Uses zero-filled data instead of random.  
- `-u, --unique_files`: Stamps every 512-byte block with (file, block, worker) so no two blocks of the run are identical, even with `-z`. By default each worker writes the same buffer to all its files, which storage with deduplication (ZFS `dedup=on`, NetApp, Pure, VAST, ...) can store once and report as optimistic. iotest warns when a target directory is on ZFS with dedup enabled or on NFS/SMB. Stamping adds a few ms per 30 MiB file to sync times.  
- `--verify`: Makes every test file self-checking, for storage acceptance tests. Each file ends with 4 bytes holding the CRC-32 (little-endian, as `zlib.crc32` computes it) of all the bytes before them, and every read (`read`, `random`, `comprehensive` and the `simultaneous` readers) checks it, in this run or any later one, without iotest storing anything. A mismatch fails that read with `verify failed: <path>: CRC-32 mismatch (stored 0x…, computed 0x…)`, so it counts in the failed ops and iotest exits 1. This detects bit flips and torn or misplaced writes (a write that only partly reached the media, or data that landed in another file or at another offset): `--verify` implies `-u`, so no two blocks are the same and a block in the wrong place changes the CRC. A file is checked only against itself, so a whole write that was lost while an older, intact file of the same name stayed is not detected. Files written without `--verify` have no CRC and fail the check: write them with `--verify` too before reading them with it. Needs a file size of at least 28 bytes (exit 2 otherwise). CPU cost: one CRC-32 pass over every byte written and read (about 8 GiB/s per core with zlib 1.3; it depends on the zlib build and the CPU). Without `-bs` it runs outside the op time, before each timed write and after each timed read, so only the sync figures include it; with `-bs` it runs block by block inside the timed op, like `-u` stamping.  
- `-cl, --cleanup`: Deletes the test files when the run ends. By default they are kept so a later `read` run can reuse them (the default settings leave about 53 GiB). Before writing, iotest checks free space per filesystem and refuses to start (exit 1) if the run would not fit.  
- `-nl, --no_log`: Disables log file creation.  
- `-addr, --message_end_point_address`: Posts a short report of each result to this URL, as Discord-style JSON (`{"content": "..."}`) with a `POST`. A failure to post is logged and does not change the exit code.  
- `--threshold_to_report_anomaly`: Adds a warning to the report when a mode's 99th-percentile latency is more than this many times its 1st-percentile latency (default: 0, off).  
- `-bt, --benchmark_time`: Seconds to spend benchmarking test data generation speed before the run (default: 5).  
- `-nr, --no_report`: Disables result report creation.  
- `-tl, --timeline`: Also writes the [per-operation timeline](#per-operation-timeline), `<name>_ops.csv`. Off by default; with `-tl` it is written even with `-nr` or `-S`. Costs memory per operation (see the timeline section).  
- `-j, --json PATH`: Writes a JSON summary of the run to `PATH` after the text report, also with `-nr`. `-` prints it to stdout instead; with `-S`, stdout then carries only the JSON (worker progress digits go to stderr). With throughput tuning it describes the final run. A summary that cannot be written exits 1, like other result files.  

### Modes
- `write` / `w`: Only file writes (default).  
- `read` / `r`: Only file reads.  
- `index` / `i`: Create and remove temporary index folders.  
- `meta` / `m`: Metadata rates, like mdtest. Three passes in order, each with its own start barrier: create, stat, then unlink `-fc` empty files per process (under `<directory>/<worker>/meta/`, e.g. `<directory>/0000/meta/`, as `meta_<i>`). Results are under the keys `meta_create`, `meta_stat` and `meta_unlink`, and the timeline's `mode` is `meta` with `op` `create`, `stat` or `unlink`. They have IOPS and latency but no bandwidth, since no data is moved; `-fs` and `-bs` do not apply and no memory is reserved for file content. Leftover meta files of an interrupted run are removed (untimed) before the create pass, the empty meta directories (and their counter directories, if nothing else is in them) are removed after the unlink pass, and `-cl` removes any left behind. A file missing in the stat or unlink pass counts as a failed operation.  
- `benchmark` / `b`: Only measures how fast the test data can be generated; no test files are written.  
- `random`: Random read/write steps. Reads and writes are reported separately, as `random_write` and `random_read` (and as separate CSV columns); both use the whole pass's wall time for the sync figures.  
- `comprehensive` / `c`: write → move → stat → read per file. Each section ends with a note that its sync figures are estimates: the pass time is split across write/move/stat/read by their share of op time.  
- `rw` / `wr`: Run write, then read sequentially in batch mode.  
- `simultaneous` / `s`: Pre-writes read targets, then runs half the workers writing and half reading concurrently on separate files. Removes all temp files when done. If the layout pass cannot write every read file (including all the files of a layout worker that dies), the timed pass is skipped and the failures are reported under the result key `layout` (`layout_1`, `layout_2`, ... when several passes fail): in the report as `Failed layout ops: <failed> of <read files>`, in the CSV as an empty `layout_time` column, in the JSON and the short reports. Layout writes are not timed, so `layout` has no times.  
- `rwi` / `wri`: Run write, then read, then index sequentially in batch mode.

Example:
```bash
iotest -fs 50m -fc 100 -pc 4 -d /tmp/iotest write read
```
Will launch 4 processes each write same random 50MiB size data to 100 separate files sequentially.
Then will launch another 4 processes reading the same files.

Simultaneous read/write example:
```bash
iotest -fs 50m -fc 20 -pc 8 -d /tmp/iotest simultaneous
```

### Test files
Each worker keeps its files in a directory of its own, named by its number padded to 4 digits whatever the process count: `<directory>/0000/temp_<i>.bin`, `<directory>/0001/temp_<i>.bin`, ... (`<i>` is padded to the width of `-fc`). With several `-d` directories, worker `n` uses directory `n` modulo their count. So runs at different process counts share files: a `-pc 9` read reads the files a `-pc 12` write left in `0000` to `0008`, and `-cl` and the leftover hint cover only the workers the run used. `meta` uses `<directory>/0000/meta/` and so on, and `simultaneous` uses `<directory>/.iotest_simul_<datetime>/worker_0000/` and so on.

Upgrading from 3.69 or earlier: worker directories used to be padded to the width of the process count (`0`, `1`, ... or `00`, `01`, ...). Test files left in those directories are not reused (a `read` run recreates its files in `0000`, ...) and not removed by `-cl`: delete them by hand.

### Measurement notes
- Page cache: after each write is synced, iotest drops the file's pages (`posix_fadvise(DONTNEED)`) and before each read it does the same, so results reflect the storage. On macOS, which has no `posix_fadvise`, `F_NOCACHE` is set on every test file instead. On Windows neither exists, so reads may be served from cache.
- Start delay: each pass starts at a common time, far enough ahead for the workers to start, make their file content (timed by the generation benchmark) and allocate and zero-fill their read buffers, before the timed pass. The buffer time is estimated from one allocation of up to 64 MiB, assuming parallel workers together go at most twice as fast as one; the `Read buffer alloc time` and `Process start delay` lines show the estimate.
- Late starts: workers wait for a common start time. If some reach it late (for example, a slow start under heavy load), the report adds a line such as `Late starts:  3 of 36 workers, up to 0.0412 s late (sync figures include the delay)` and a warning is logged, even without `-v`. The returned result also carries them as `late_starts`.

### Per-operation timeline
With `-tl/--timeline`, iotest also writes `<name>_ops.csv` (next to the other result files, in the first `-d` directory), one row per attempted operation, so you can see SSD cache exhaustion, throttling or stalls over time. It is off by default, and `-tl` writes it even with `-nr` or `-S`. Columns: `pass` (1-based index of the pass in this run; a simultaneous run is one pass), `mode` (the pass's mode; the three passes of a `meta` run all say `meta`), `op` (`write`, `read`, `move`, `stat`, `index`, `create` or `unlink`), `worker` (the worker's name, as used for its directory: `0000`, `0001`, ...), `file_index` (1-based), `start_s` (seconds from the pass's common start time to the operation's start, 6 decimals; for a failed operation, when the attempt began), `duration_s` (6 decimals, empty for a failed operation) and `error` (empty on success, the error text on failure). Rows are sorted by pass, then `start_s`. Workers that died before reporting have no rows. Throughput tuning attempts do not record it; the final tuning run does.

Recording it does no I/O and next to no work in the timed passes: each worker appends every operation to compact arrays (21 bytes an operation) and hands them over once, after its last operation. iotest keeps them until the run ends and writes the file after the report; with the transfers in between, that takes about 70 (meta) to 110 bytes of memory per operation, on top of the 80 to 140 every run needs for its results, and the memory check counts both. For example, `-pc 64 -fc 100000 meta` (19.2 million operations) is estimated at about 1.5 GB for its results and 1.3 GB more with `-tl`.

### JSON summary
`--json` is meant for collecting runs from many hosts (for example launched with `mssh`). Top level: `tool`, `version`, `host`, `started` (local time with UTC offset), `options` (`file_size`, `file_count`, `process_count` as requested, `directories`, `modes`, `zeros`, `unique` (also `true` with `--verify`), `start_at`, `block_size` as requested or `null`, `verify`, `timeline`), `start_late_s` (seconds the synchronized first pass started after `--start_at`, `0.0` when on time, `null` without `--start_at`), `results`, `failed_ops` and `report_errors`. `results` has one entry per result key (`write`, `read`, `random_write`, ..., and `layout` for a simultaneous layout pass that failed, with `ops` 0 and `failed` the read files it could not write), each with `ops`, `failed`, `first_error`, `process_count` (the count that ran, which the memory check can lower), `total_bytes`, `bytes_per_s_call`, `bytes_per_s_sync`, `io_time_pct`, `iops_call`, `iops_sync`, `latency_s` (`avg`, `median`, `p1`, `p99`, `p99_9`, `max`, in seconds; `null` when no op succeeded), `sync_estimated`, `late_workers` and `max_late_s`. Values that do not apply (bytes for `index`, for example) are `null`.

### Multi-host runs
`--start_at EPOCH` makes every host begin its first timed pass at the same wall-clock instant (seconds since the Unix epoch, e.g. `$(date -d '+1 min' +%s)`), so the storage sees the combined load. Compute the time once, locally, and pass it to every host:
```bash
T=$(date -d '+2 min' +%s)
mssh node[1-8] "iotest -d /scratch/\$(hostname) --start_at $T -j /scratch/iotest-\$(hostname).json"
```
- Each host needs its own `-d` directory (as `$(hostname)` above). On a shared directory the hosts' worker files collide, their result files can get identical names (every part of the name, including the datetime to the second, can match across a fan-out) and overwrite each other, and simultaneous mode's `.iotest_simul_<datetime>` trees collide.
- The hosts' clocks must be in sync (NTP/chrony); iotest cannot correct for skew.
- Only the first timed pass is synchronized (for `simultaneous`, the pass after its layout pass); later passes start as usual.
- The time must be in the future when iotest parses its arguments, otherwise it exits with `2`. Leave enough time for the file generation benchmark; if less than the usual process start delay remains when the pass is scheduled, it starts after that delay instead, a warning says how late, and the report notes it.
- It cannot be combined with `-wtpt`/`-rtpt`.
- iotest prints `Waiting until <local time> for --start_at (<N.N> s)` while it waits. The report starts with `Synchronized start at <local time> (--start_at)`, and the JSON `options` carry `start_at` and the top level `start_late_s`.

### Exit codes
- `0`: the run completed and every operation succeeded.
- `1`: at least one I/O operation failed (the report shows how many and the first error; a `--verify` mismatch counts as a failed read), the target does not have enough free space (nothing is written), throughput tuning could not reach its target, `simultaneous` could not write its read files (that pass's timed part does not run; the failures count as failed `layout` ops), or a result file could not be written (the report is still printed).
- `2`: invalid arguments, e.g. `-fc 0` or an unparseable `-fs`.

Check the available arguments with `-h` or `--help` for more details.

```bash
$ iotest -h
usage: iotest [-h] [-fs FILE_SIZE] [-bs BLOCK_SIZE] [-fc FILE_COUNT] [-t PROCESS_COUNT] [-ts TOTAL_SIZE]
              [-d DIRECTORY] [-ld LOG_DIRECTORY] [-q] [-v] [-S] [-nl] [-nr] [-tl] [-z] [-u] [--verify] [-j PATH] [-cl]
              [-addr MESSAGE_END_POINT_ADDRESS] [--threshold_to_report_anomaly THRESHOLD_TO_REPORT_ANOMALY]
              [-bt BENCHMARK_TIME] [-wtpt WRITE_TARGET_THROUGHPUT] [-rtpt READ_TARGET_THROUGHPUT] [--start_at EPOCH]
              [-V]
              [{read,write,random,index,meta,m,benchmark,comprehensive,simultaneous,r,w,rw,wr,i,rwi,wri,c,b,s} ...]

Test total disk bandwidth. Default mode is write (w). Comprehensive mode: write -> move -> stat -> read

positional arguments:
  {read,write,random,index,meta,m,benchmark,comprehensive,simultaneous,r,w,rw,wr,i,rwi,wri,c,b,s}
                        The mode the script will operate in (default: write / w). COMPREHENSIVE: per-thread write ->
                        move -> stat -> read (what your code sees). WRITE: batched all-thread write. READ: batched
                        all-thread read. INDEX: creates --file_count index folders, stat each, then delete. META / M:
                        create, stat, then unlink --file_count empty files per process, each as its own timed pass
                        (keys meta_create, meta_stat, meta_unlink); -fs does not apply. RWI / WRI: run write, then
                        read, then index sequentially in batch mode. SIMULTANEOUS / S: layout write pass for read
                        files, then concurrent write+read on separate files; cleans up afterward.

options:
  -h, --help            show this help message and exit
  -fs, --file_size FILE_SIZE
                        File size (default:30), defaults to mb, can specify in t(b),g(b),m(b),k(b),b
  -bs, --block_size BLOCK_SIZE
                        Write and read each file in calls of this size instead of the whole file in one call
                        (default): each worker reads into one buffer of this size and writes from a pool of 16 MiB of
                        test data and one block (or the file size), so the file does not repeat within any 16 MiB.
                        Same units as -fs
  -fc, --file_count FILE_COUNT
                        Number of files to create and read per process (default:50)
  -t, -pc, --process_count PROCESS_COUNT
                        Number of processes to run concurrently (default:36)
  -ts, --total_size TOTAL_SIZE
                        Total size target. If -fs/-fc/-pc is omitted, auto-calculate one missing arg to fit this total
                        size.
  -d, --directory DIRECTORY
                        Directory to put the files in. Repeat -d to round-robin workers across directories
                        (default:<pwd>)
  -ld, --log_directory LOG_DIRECTORY
                        Directory to put the log files in (default:/var/log/)
  -q, --quiet           Kept for compatibility: quiet is the default. Use -v for per-file output
  -v, --verbose         Verbose output
  -S, --stealth         Suppress verbose output and verbose log file
  -nl, --no_log         Do not write log files
  -nr, --no_report      Do not write report files
  -tl, --timeline       Also write <report name>_ops.csv, the start time, duration and error of every operation, even
                        with -nr or -S. Costs up to about 110 bytes of memory per operation
  -z, --zeros           Use zeros instead of random numbers. Use this if you are sure no write compression is
                        available. Potentially higher write accuracy.
  -u, --unique_files    Stamp every 512-byte block with (file, block, worker) so no two blocks of the run are
                        identical. Use on storage that deduplicates (iotest warns when it detects some). Adds a few ms
                        per 30 MiB file to sync times.
  --verify              End every file with a CRC-32 of its other bytes and check it on every read, in this run or a
                        later one, to detect bit flips and torn or misplaced writes. Implies -u. Needs a file size of
                        at least 28 bytes; files written without --verify fail the check
  -j, --json PATH       Write a JSON summary of the run to PATH after the text report. Use - for stdout (with -S,
                        stdout then carries only the JSON)
  -cl, --cleanup        Delete the test files when the run ends (default: keep them, so a later read run can reuse
                        them)
  -addr, --message_end_point_address MESSAGE_END_POINT_ADDRESS
                        The end point address of the message
  --threshold_to_report_anomaly THRESHOLD_TO_REPORT_ANOMALY
                        The threshold to report if 1 percent high is higher then 1 percent low *
                        <threshold_to_report_anomaly>
  -bt, --benchmark_time BENCHMARK_TIME
                        The time in seconds to run the benchmark for file generation speed
  -wtpt, --write_target_throughput WRITE_TARGET_THROUGHPUT
                        Target sync write throughput (e.g. 500m, 2g). Auto-tunes the first write pass.
  -rtpt, --read_target_throughput READ_TARGET_THROUGHPUT
                        Target sync read throughput (e.g. 500m, 2g). Auto-tunes the first read pass.
  --start_at EPOCH      Start the first timed pass at this time, in seconds since the Unix epoch (e.g. $(date -d '+1
                        min' +%s)), so several hosts start together. Clocks must be in sync
  -V, --version         show program's version number and exit
```
