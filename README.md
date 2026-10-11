# iotest

Ever feel like fio is too complicated and too optimized? Well, simple-iotest covers this for you!

Using the most simple non-optimized write() or writev() with fallback to python f.write() (just like your code!) to test file system io performance.

Warning: if using big file sizes, make sure you have enough memory to hold all these bits!

Note:
	Recommand to also install the package Tee_Logger to also log your test results to your /var/log/ ( configurable ) for future keeping.

Generated:

This script tests I/O performance by creating, reading, moving, and indexing files in various modes. It allows you to measure different aspects of disk performance and log the results.

## Installation
```bash
pipx install simple-iotest
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
- `-fc, --file_count`: Number of files to process per worker (default: 50).  
- `-pc, --process_count`: Number of worker processes (default: 36).  
- `-wtpt, --write_target_throughput`: Target sync write throughput (e.g. `500m`, `2g`); auto-tunes the first write pass.  
- `-rtpt, --read_target_throughput`: Target sync read throughput; auto-tunes the first read pass. For simultaneous mode with both targets, the initial writer/reader split is estimated from the target ratio before tuning.  
- `-d, --directory`: Directory for file operations.  
- `-q, --quiet`: Suppresses output.  
- `-z, --zeros`: Uses zero-filled data instead of random.  
- `-u, --unique_files`: Stamps every 512-byte block with (file, block, worker) so no two blocks of the run are identical, even with `-z`. By default each worker writes the same buffer to all its files, which storage with deduplication (ZFS `dedup=on`, NetApp, Pure, VAST, ...) can store once and report as optimistic. iotest warns when a target directory is on ZFS with dedup enabled or on NFS/SMB. Stamping adds a few ms per 30 MiB file to sync times.  
- `-cl, --cleanup`: Deletes the test files when the run ends. By default they are kept so a later `read` run can reuse them (the default settings leave about 53 GiB). Before writing, iotest checks free space per filesystem and refuses to start (exit 1) if the run would not fit.  
- `-nl, --no_log`: Disables log file creation.  
- `-nr, --no_report`: Disables result report creation.  

### Modes
- `write` / `w`: Only file writes (default).  
- `read` / `r`: Only file reads.  
- `index` / `i`: Create and remove temporary index folders.  
- `random`: Random read/write steps.  
- `comprehensive` / `c`: write → move → stat → read per file.  
- `rw` / `wr`: Run write, then read sequentially in batch mode.  
- `simultaneous` / `s`: Pre-writes read targets, then runs half the workers writing and half reading concurrently on separate files. Removes all temp files when done.  
- `rwi` / `wri`: Run write, then read, then index sequentially in batch mode.

Example:
```bash
iotest -fs 50m -fc 100 -pc 4 -d /tmp/iotest write read
```
Will launch 4 processes each write same random 50MiB size data to 100 seperate files sequentially.
Then will launch another 4 processes reading the same files.

Simultaneous read/write example:
```bash
iotest -fs 50m -fc 20 -pc 8 -d /tmp/iotest simultaneous
```

### Exit codes
- `0`: the run completed and every operation succeeded.
- `1`: at least one I/O operation failed (the report shows how many and the first error), the target does not have enough free space (nothing is written), or throughput tuning could not reach its target.
- `2`: invalid arguments, e.g. `-fc 0`.

Check the available arguments with `-h` or `--help` for more details.

```bash
$ iotest -h
usage: iotest [-h] [-fs FILE_SIZE] [-fc FILE_COUNT] [-t PROCESS_COUNT] [-ts TOTAL_SIZE] [-d DIRECTORY]
              [-ld LOG_DIRECTORY] [-q] [-v] [-S] [-nl] [-nr] [-z] [-u] [-cl] [-addr MESSAGE_END_POINT_ADDRESS]
              [--threshold_to_report_anomaly THRESHOLD_TO_REPORT_ANOMALY] [-bt BENCHMARK_TIME]
              [-wtpt WRITE_TARGET_THROUGHPUT] [-rtpt READ_TARGET_THROUGHPUT] [-V]
              [{read,write,random,index,benchmark,comprehensive,simultaneous,r,w,rw,wr,i,rwi,wri,c,b,s} ...]

Test total disk bandwidth. Default mode is write (w). Comprehensive mode: write -> move -> stat -> read

positional arguments:
  {read,write,random,index,benchmark,comprehensive,simultaneous,r,w,rw,wr,i,rwi,wri,c,b,s}
                        The mode the script will operate in (default: write / w). COMPREHENSIVE: per-thread write ->
                        move -> stat -> read (what your code sees). WRITE: batched all-thread write. READ: batched
                        all-thread read. INDEX: creates --file_count index folders, stat each, then delete. RWI / WRI:
                        run write, then read, then index sequentially in batch mode. SIMULTANEOUS / S: layout write
                        pass for read files, then concurrent write+read on separate files; cleans up afterward.

options:
  -h, --help            show this help message and exit
  -fs, --file_size FILE_SIZE
                        File size (default:30), defaults to mb, can specify in t(b),g(b),m(b),k(b),b
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
  -z, --zeros           Use zeros instead of random numbers. Use this if you are sure no write compression is
                        available. Potentially higher write accuracy.
  -u, --unique_files    Stamp every 512-byte block with (file, block, worker) so no two blocks of the run are
                        identical. Use on storage that deduplicates (iotest warns when it detects some). Adds a few ms
                        per 30 MiB file to sync times.
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
  -V, --version         show program's version number and exit
```
