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
```

### Common Options
- `-fs, --file_size`: Size of the test files (can include suffix like `m`, `g`, etc.).  
- `-fc, --file_count`: Number of files to process per worker.  
- `-pc, --process_count`: Number of worker processes.  
- `-d, --directory`: Directory for file operations.  
- `-q, --quiet`: Suppresses output.  
- `-z, --zeros`: Uses zero-filled data instead of random.  
- `-nl, --no_log`: Disables log file creation.  
- `-nr, --no_report`: Disables result report creation.  

### Modes
- `write` / `w`: Only file writes.  
- `read` / `r`: Only file reads.  
- `index` / `i`: Create and remove temporary index folders.  
- `random`: Random read/write steps.  
- `comprehensive` / `c`: Includes write, index, read, etc.  
- `rw`: Do write → read in the same operation
- `rwi` or `wri`: Do write → index → read in the same operation.

Example:
```bash
iotest -fs 50m -fc 100 -pc 4 -d /tmp/iotest write read
```
Will launch 4 processes each write same random 50MiB size data to 100 seperate files sequentially.
Then will launch another 4 processes reading the same files.

Check the available arguments with `-h` or `--help` for more details.

```bash
$ iotest -h
usage: iotest [-h] [-fs FILE_SIZE] [-fc FILE_COUNT] [-t PROCESS_COUNT] [-d DIRECTORY] [-ld LOG_DIRECTORY] [-q] [-v] [-S]
              [-nl] [-nr] [-z] [-addr MESSAGE_END_POINT_ADDRESS] [--threshold_to_report_anomaly THRESHOLD_TO_REPORT_ANOMALY]
              [-V]
              [{comprehensive,read,write,random,index,r,w,rw,wr,i,rwi,wri,c} ...]

Test total disk bandwidth. Default to comprehensive mode: write -> move -> stat -> read

positional arguments:
  {comprehensive,read,write,random,index,r,w,rw,wr,i,rwi,wri,c}
                        The mode the script will operate in (default:comprehensive). COMPREHENSIVE: async fully cached per thread write
                        - index - read operation. WRITE: batched all thread write. READ: batched all thread read. INDEX: creates
                        --file_count amount of index folders, stat it, then delete it. RWI: Execute Write - Index - Read mode
                        sequentially in batch mode.

options:
  -h, --help            show this help message and exit
  -fs, --file_size FILE_SIZE
                        File size (default:30), defaults to mb, can specify in t(b),g(b),m(b),k(b),b
  -fc, --file_count FILE_COUNT
                        Number of files to create and read per process (default:50)
  -t, -pc, --process_count PROCESS_COUNT
                        Number of processes to run concurrently (default:27)
  -d, --directory DIRECTORY
                        Directory to put the files in (default:<pwd>)
  -ld, --log_directory LOG_DIRECTORY
                        Directory to put the log files in (default:/var/log/)
  -q, --quiet           Suppress output, default True in new version
  -v, --verbose         Verbose output
  -S, --stealth         Suppress verbose output and verbose log file
  -nl, --no_log         Do not write log files
  -nr, --no_report      Do not write report files
  -z, --zeros           Use zeros instead of random numbers. Use this if you are sure no write compression is available. Potentially
                        higher write accuracy.
  -addr, --message_end_point_address MESSAGE_END_POINT_ADDRESS
                        The end point address of the message
  --threshold_to_report_anomaly THRESHOLD_TO_REPORT_ANOMALY
                        The threshold to report if 1 percent high is higher then 1 percent low * <threshold_to_report_anomaly>
  -V, --version         show program's version number and exit
```