#!/usr/bin/env python3
import argparse
import os
import random
import time
from multiprocessing import Process, Manager
import datetime
import subprocess
import json

try:
    from memory_profiler import profile
except:
    def profile(func):
        return func

version = '3.59.4'
__version__ = version

# --------------------------------
# TeeLogger Inline print only
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    warning = '\033[93m'
    critical = '\033[91m'
    info = '\033[0m'
    debug = '\033[0m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printWithColor(msg, level = 'info'):
    if level == 'info':
        print(msg)
    elif level == 'debug':
        print(bcolors.debug + msg + bcolors.ENDC)
    elif level == 'warning':
        print(bcolors.warning + msg + bcolors.ENDC)
    elif level == 'error':
        print(bcolors.warning + msg + bcolors.ENDC)
    elif level == 'critical':
        print(bcolors.critical + msg + bcolors.ENDC)
    elif level == 'ok' or level == 'okgreen':
        print(bcolors.OKGREEN + msg + bcolors.ENDC)
    elif level == 'okblue':
        print(bcolors.OKBLUE + msg + bcolors.ENDC)
    elif level == 'okcyan':
        print(bcolors.OKCYAN + msg + bcolors.ENDC)
    else:
        print(bcolors.info + msg + bcolors.ENDC)
try:
    import Tee_Logger
except:
    class Tee_Logger:
        version = '0.1 inline'
        class teeLogger:
            def __init__(self, systemLogFileDir='.', programName='iotest', compressLogAfterMonths=2, deleteLogAfterYears=2, suppressPrintout=False, fileDescriptorLength=15,noLog=True):
                self.name = programName
                self.currentDateTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.noLog = True
                if not noLog:
                    print('Using inline logger, foring noLog...')
                self.systemLogFileDir = '/dev/null'
                self.logsDir = '/dev/null'
                self.logFileDir = '/dev/null'
                self.logFileName = '/dev/null'
                self.compressLogAfterMonths = compressLogAfterMonths
                self.deleteLogAfterYears = deleteLogAfterYears
                self.suppressPrintout = suppressPrintout
                self.fileDescriptorLength = fileDescriptorLength
                self.version = Tee_Logger.version

                self.logger = None

            def log_with_caller_info(self, level, msg):
                return

            def teeok(self, msg):
                if not self.suppressPrintout:
                    printWithColor(msg, 'okgreen')
                self.log_with_caller_info('info', msg)

            def ok(self, msg):
                self.log_with_caller_info('info', msg)

            def teeprint(self, msg):
                if not self.suppressPrintout:
                    printWithColor(msg, 'info')
                self.log_with_caller_info('info', msg)

            def info(self, msg):
                self.log_with_caller_info('info', msg)

            def teeerror(self, msg):
                if not self.suppressPrintout:
                    printWithColor(msg, 'error')
                self.log_with_caller_info('error', msg)

            def error(self, msg):
                self.log_with_caller_info('error', msg)

            def teelog(self, msg, level):
                if not self.suppressPrintout:
                    printWithColor(msg, level)
                self.log_with_caller_info(level, msg)

            def log(self, msg, level):
                self.log_with_caller_info(level, msg)

# ------------------------------
def format_bytes(size, use_1024_bytes=True):
    # size is in bytes
    if use_1024_bytes:
        power = 2**10
        n = 0
        power_labels = {0 : '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti', 5: 'Pi'}
        while size > power:
            size /= power
            n += 1
    else:
        power = 10**3
        n = 0
        power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T', 5: 'P'}
        while size > power:
            size /= power
            n += 1
    return f"{size:.2f} {power_labels[n]}"

def almost_urandom(n):
    try:
        return random.getrandbits(8 * n).to_bytes(n, 'big')
    except OverflowError:
        return almost_urandom(n // 2) + almost_urandom(n - n // 2)
    
def create_file(file_name, file_content,file_size,quiet=False,tl=None):
    if not tl:
        tl = Tee_Logger.teeLogger(suppressPrintout=quiet)
    try:
        with open(file_name, "wb", buffering=0) as f:
            try:
                if os.name == 'posix':
                    os.posix_fadvise(f.fileno(), 0, file_size, os.POSIX_FADV_DONTNEED)
            except:
                tl.teelog(f'Failed to posix_fadvise, trying fallocate',level='warning')
            start_write_time = time.perf_counter()
            try:
                if os.name == 'posix':
                    os.writev(f.fileno(), [file_content])
                else:
                    os.write(f.fileno(), file_content)
            except:
                tl.teelog(f'Failed to write using os.writev, trying f.writeinto',level='warning')
                f.write(file_content)
            os.fsync(f.fileno())
            end_write_time = time.perf_counter()
            return start_write_time,end_write_time
    except Exception as e:
        import traceback
        tl.teeerror(str(e))
        tl.teeerror(traceback.format_exc())
        return 0,time.perf_counter()

def move_file(src, dst,tl=None):
    if not tl:
        tl = Tee_Logger.teeLogger()
    start_move_time = time.perf_counter()
    try:
        os.rename(src, dst)
        end_move_time = time.perf_counter()
        return start_move_time,end_move_time
    except Exception as e:
        import traceback
        tl.teeerror(str(e))
        tl.teeerror(traceback.format_exc())
        return 0,time.perf_counter()

def read_file(file_name,file_content, file_size,quiet=False,tl=None):
    if not tl:
        tl = Tee_Logger.teeLogger(suppressPrintout=quiet)
    b=bytearray(file_size)
    # check if file exists and size is correct
    try:
        if not os.path.isfile(file_name) or os.path.getsize(file_name) != file_size:
            # file does not exist or size is wrong, create it
            if not quiet:
                tl.teeerror(f"File {file_name} does not exist or size is wrong, creating it...")
            create_file(file_name, file_content,file_size,tl=tl)

        with open(file_name, "rb", buffering=0) as f:
            try:
                if os.name == 'posix':
                    os.posix_fadvise(f.fileno(), 0, file_size, os.POSIX_FADV_DONTNEED)
            except:
                tl.teelog(f'Failed to posix_fadvise, trying fallocate',level='warning')
            start_read_time = time.perf_counter()
            try:
                if os.name == 'posix':
                    os.readv(f.fileno(),[b])
                else:
                    f.readinto(b)
            except:
                tl.teelog(f'Failed to read using os.readv, trying f.readinto',level='warning')
                f.readinto(b)
            end_read_time = time.perf_counter()
            return start_read_time,end_read_time
    except Exception as e:
        import traceback
        tl.teeerror(str(e))
        tl.teeerror(traceback.format_exc())
        return 0,time.perf_counter()
    

def index_file(file_name,quiet=False,tl=None):
    if not tl:
        tl = Tee_Logger.teeLogger(suppressPrintout=quiet)
    try:
        # index creates file_name.index folder, stat it, then delete it
        # create the index folder
        index_folder = file_name + '.index'
        start_index_time = time.perf_counter()
        os.makedirs(index_folder, exist_ok=True)
        # stat the index folder
        os.stat(index_folder)
        # delete the index folder
        os.rmdir(index_folder)
        end_index_time = time.perf_counter()
        return start_index_time,end_index_time

    except Exception as e:
        import traceback
        tl.teeerror(str(e))
        tl.teeerror(traceback.format_exc())
        return 0,time.perf_counter()
    
def stat_file(file_name,tl=None):
    if not tl:
        tl = Tee_Logger.teeLogger()
    start_stat_time = time.perf_counter()
    try:
        size = os.stat(file_name).st_size
        end_stat_time = time.perf_counter()
        return start_stat_time,end_stat_time,size
    except Exception as e:
        import traceback
        tl.teeerror(str(e))
        tl.teeerror(traceback.format_exc())
        return 0,time.perf_counter(),0

def int_to_color(n, brightness_threshold=500):
    hash_value = hash(str(n))
    r = (hash_value >> 16) & 0xFF
    g = (hash_value >> 8) & 0xFF
    b = hash_value & 0xFF
    if (r + g + b) < brightness_threshold:
        return int_to_color(hash_value, brightness_threshold)
    return (r, g, b)

def worker(file_count, file_size, directory, results, mode, counter,quiet,zeros,thread_start_time,tl=None):
    if not tl:
        tl = Tee_Logger.teeLogger(suppressPrintout=quiet)
    local_results = []
    r, g, b = int_to_color(os.getpid())
    if not quiet:
        tl.teeprint(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} scheduled to start at {thread_start_time-time.perf_counter():.4f} later in {mode} mode.' + '\033[0m')
    if zeros:
        file_content = b'\x00' * file_size
    else:
        file_content = almost_urandom(file_size)
    if time.perf_counter() > thread_start_time:
        tl.teeerror(f'Worker {counter} started late, expected start time {thread_start_time}, actual start time {time.perf_counter()}')
    if not quiet:
        tl.teeprint(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} primed, waiting for {thread_start_time-time.perf_counter():.4f} seconds.' + '\033[0m')
    # wait until thread_start_time
    while time.perf_counter() < thread_start_time:
        # sleep for 1 ms
        time.sleep(0.001)
    os.makedirs(os.path.join(directory,str(counter)), exist_ok=True)
    reportInterval = max(file_count // 10,1)
    for i in range(file_count):
        # print a digit every 10% of the files
        if i % reportInterval == 0 and quiet:
                print(f'\r{i//reportInterval}',end='',flush=True)
        i = str(i+1).zfill(len(str(file_count)))
        file_name = os.path.join(directory,str(counter), f"temp_{i}.bin")

        if mode == 'write':

            start_write_time,end_write_time = create_file(file_name, file_content,file_size,quiet=quiet,tl=tl)
            
            local_results.append(end_write_time- start_write_time)
            if not quiet:
                tl.teeprint(f'\033[38;2;{r};{g};{b}m' + f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tWrote\tin {end_write_time-start_write_time} s"+ '\033[0m') 

        elif mode == 'read':
            
            start_read_time,end_read_time = read_file(file_name, file_content, file_size,quiet,tl=tl)

            local_results.append(end_read_time- start_read_time)
            if not quiet:
                tl.teeprint(f'\033[38;2;{r};{g};{b}m' +f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tRead\tin {end_read_time-start_read_time} s"+ '\033[0m')

        elif mode == 'random':
            # here we start a random read or write
            if bool(random.getrandbits(1)):
                # if zeros:
                #     file_content = b'\x00' * file_size
                # else:
                #     file_content = almost_urandom(file_size)
                start_write_time,end_write_time = create_file(file_name, file_content,file_size,quiet=quiet,tl=tl)
                local_results.append(end_write_time- start_write_time)
                if not quiet:
                    tl.teeprint(f'\033[38;2;{r};{g};{b}m' + f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tWrote\tin {end_write_time-start_write_time} s"+ '\033[0m')
            else:
                start_read_time,end_read_time = read_file(file_name,file_content, file_size,quiet,tl=tl)
                local_results.append(end_read_time- start_read_time)
                if not quiet:
                    tl.teeprint(f'\033[38;2;{r};{g};{b}m' +f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tRead\tin {end_read_time-start_read_time} s"+ '\033[0m')
        elif mode == 'comprehensive':
            resultList = []
            # first, we write the file
            # if zeros:
            #     file_content = b'\x00' * file_size
            # else:
            #     file_content = almost_urandom(file_size)

            start_write_time,end_write_time = create_file(file_name, file_content,file_size,quiet=quiet,tl=tl)
            
            resultList.append(end_write_time- start_write_time)

            # then we move the file
            start_move_time,end_move_time = move_file(file_name, file_name + ".moved",tl=tl)

            resultList.append(end_move_time- start_move_time)

            # then we stat the file
            start_stat_time,end_stat_time,size = stat_file(file_name + ".moved",tl=tl)

            if size != file_size:
                tl.teeerror(f'{bcolors.critical}File {file_name} size is wrong, expected {file_size}, got {size}\033[0m')
                #exit(1)

            resultList.append(end_stat_time- start_stat_time)

            # then we read the file

            start_read_time,end_read_time = read_file(file_name + ".moved",file_content, file_size,quiet,tl=tl)

            resultList.append(end_read_time- start_read_time)

            if not quiet:
                #print(f'\033[38;2;{r};{g};{b}m' + f"[Process {os.getpid()}]\tFile {i + 1}/{file_count}:\t{file_name}\tWrote\tin {end_write_time-start_write_time} s"+ '\033[0m') 
                tl.teeprint(f'\033[38;2;{r};{g};{b}m[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tW: {resultList[0]:.4f} s M: {resultList[1]:.4f} s S: {resultList[2]:.4f} s R: {resultList[3]:.4f} s\033[0m')
            local_results.append(resultList)
        elif mode == 'index':
            # This will test the indexing performance of the filesystem
            start_index_time,end_index_time = index_file(file_name, quiet,tl=tl)
            local_results.append(end_index_time- start_index_time)
            if not quiet:
                tl.teeprint(f'\033[38;2;{r};{g};{b}m' +f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tIndexed\tin {end_index_time-start_index_time} s"+ '\033[0m')

    if not quiet:
        tl.teeprint(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} finished.' + '\033[0m')
    results.extend(local_results)


# This code adds a key to a dictionary, 
# but if the key already exists, 
# it adds a suffix to the key and tries to add it again. 
# If that key already exists, it adds another suffix and tries again, 
# until it finds a key that doesn't already exist, 
# and adds the key and value pair to the dictionary.

def addToDicWithoutOverwrite(dic,key,value):
    if key in dic:
        dic[f'{key}_1'] = dic.pop(key)
    if f'{key}_1' in dic:
        i = 2
        while f'{key}_{i}' in dic:
            i += 1
        dic[f'{key}_{i}'] = value
    else:
        dic[key] = value

def benchmarkGenSpeed(file_size, file_count,zeros,results):
    start_time = time.perf_counter()
    for i in range(file_count):
        if zeros:
            _ = b'\x00' * file_size
        else:
            _ = almost_urandom(file_size) 
        if time.perf_counter() - start_time > 5:
            break
    genTime = time.perf_counter() - start_time
    genSize = file_size * (i+1)
    genSpeed = genSize / genTime 
    results.append(genSpeed)

def main(file_size, file_count, process_count, directory,modes,quiet,zeros,tl=None,stealth=False,message_end_point_address=None,no_report=False,threshold_to_report_anomaly = 0):
    if stealth:
        quiet = True
        no_report = True
    if not tl:
        tl = Tee_Logger.teeLogger(suppressPrintout=quiet)
    os.makedirs(directory, exist_ok=True)
    processes = []
    file_size = int(file_size)

    outResults = dict()
    totalTime = {}
    estimatedTotalMemory = file_size * process_count * 1.2
    phyFreeMemory = -1
    swapMemory = -1
    if os.path.exists('/proc/meminfo'):
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        phyFreeMemory = int(line.split()[1]) * 1024
                    if 'SwapFree' in line:
                        swapMemory = int(line.split()[1]) * 1024
        except:
            tl.teeerror(f"Failed to read /proc/meminfo")
    # warn if the estimated total memory is more than 90% of the total free memory
    if phyFreeMemory + swapMemory > 0 and estimatedTotalMemory > phyFreeMemory + swapMemory:
        tl.teeerror(f"Estimated total memory usage is more than the all available memory to use.")
        tl.teeerror(f"Physical available memory: {format_bytes(phyFreeMemory)}B")
        tl.teeerror(f"Swap available memory: {format_bytes(swapMemory)}B")
        tl.teeerror(f"Estimated total memory usage: {format_bytes(estimatedTotalMemory)}B")
        process_count = int((phyFreeMemory + swapMemory ) // file_size // 1.2)
        tl.teeerror(f"Reducing the number of processes to {process_count}")
    estimatedTotalMemory = file_size * process_count * 1.2
    if phyFreeMemory > 0 and estimatedTotalMemory > phyFreeMemory * 0.9:
        tl.teeerror(f"Estimated total memory usage is more than 90% of the total non swap available memory.")
        tl.teeerror(f"You may want to reduce the file size or the number of processes.")
        tl.teeerror(f"Available memory: {format_bytes(phyFreeMemory)}B")
        tl.teeerror(f"Estimated total memory usage: {format_bytes(estimatedTotalMemory)}B")
        tl.teelog(f"If continuing, iotest will likely put very heavy pressure on swap memory and may lead to system crash.",level='critical')
        tl.teelog(f"Exit now (press Ctrl+C) or iotest will continue anyway in 15 seconds...",level='critical')
        time.sleep(16)
        tl.teelog(f"Warning! Continuing anyway... ",level='warning')
    with Manager() as manager:
        # bench mark file generation performance first
        results = manager.list()
        tl.teeprint(f"Benchmarking file generation performance... with {format_bytes(file_size)}B {'zero' if zeros else 'random'} files")
        for counter in range(process_count):
            p = Process(target=benchmarkGenSpeed, args=(file_size, file_count,zeros,results))
            processes.append(p)
        genStartTime = time.perf_counter()
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        genTime = time.perf_counter() - genStartTime
        genSpeed = sum(results)
        genTimeCalc = file_size * process_count / genSpeed
        processStartDelay = + 0.005 * process_count + 1 + genTimeCalc
        tl.teeprint(f"Generation speed:      \t{format_bytes(genSpeed)}B/s")
        tl.teeprint(f"                       \t{format_bytes(genSpeed * 8,use_1024_bytes=False)}b/s")
        tl.teeprint(f"Generation test time:  \t{genTime:.4f} s")
        tl.teeprint(f"Gen time calculated:   \t{genTimeCalc:.4f} s")
        tl.teeprint(f"Process start delay:   \t{processStartDelay:.4f} s")
        for mode in modes:
            if mode == 'benchmark':
                continue
            processes = []
            results = manager.list()
            thread_start_time = time.perf_counter() + processStartDelay
            for counter in range(process_count):
                counter = str(counter).zfill(len(str(process_count)))
                p = Process(target=worker, args=(file_count, file_size, directory, results, mode, counter,quiet,zeros,thread_start_time,tl))
                processes.append(p)

            totalStartTime = thread_start_time
            for p in processes:
                p.start()
                

            for p in processes:
                p.join()
            
            totalEndTime =  time.perf_counter() - totalStartTime

            processes.clear()

            if mode == 'comprehensive':
                writes = [result[0] for result in results]
                moves = [result[1] for result in results]
                stats = [result[2] for result in results]
                reads = [result[3] for result in results]
                sumWTime = sum(writes)
                sumMTime = sum(moves)
                sumSTime = sum(stats)
                sumRTime = sum(reads)
                sumAllTime = sumWTime + sumMTime + sumSTime + sumRTime
                addToDicWithoutOverwrite(outResults,'write',writes)
                addToDicWithoutOverwrite(outResults,'move',moves)
                addToDicWithoutOverwrite(outResults,'stat',stats)
                addToDicWithoutOverwrite(outResults,'read',reads)
                addToDicWithoutOverwrite(totalTime,'write',totalEndTime * (sumWTime / sumAllTime))
                addToDicWithoutOverwrite(totalTime,'move',totalEndTime * (sumMTime / sumAllTime))
                addToDicWithoutOverwrite(totalTime,'stat',totalEndTime * (sumSTime / sumAllTime))
                addToDicWithoutOverwrite(totalTime,'read',totalEndTime * (sumRTime / sumAllTime))
            else:
                addToDicWithoutOverwrite(outResults,mode,list(results))
                addToDicWithoutOverwrite(totalTime,mode,totalEndTime)


    # write the outResults to a csv file
    if not no_report and outResults:
        dir_str_repr = directory.replace('/','-').replace('\\','-').replace(':','-').replace('--','-').replace(' ','_')
        csv_file_name = os.path.join(directory, f"iotest_{'-'.join(modes)}_{dir_str_repr}_fs={file_size}_fc={file_count}_pc={process_count}_{tl.currentDateTime}.csv")
        with open(csv_file_name, "w") as f:
            # the headers are just the keys in the dic, 
            # the values are the list of results
            # we need to transpose the list of results
            # so that we can write it to the csv file

            # first, we write the header
            f.write(','.join([key+'_time' for key in outResults.keys()]) + '\n')
            # then we write the results
            for row in zip(*outResults.values()):
                f.write(','.join(map(str,row)) + '\n')

        tl.teeprint(f"\nResults written to {csv_file_name}")

    # calculate the summary for each key in outResults
    # we need to calculate:
    # average time
    # median time
    # 1 % low time
    # 1 % high time
    # 0.1 % high time
    # highest time
    # total bandwidth
    # total size
    report = []
    for mode in outResults:
        report.append(f"*"*80)
        report.append(f"Report for {mode} mode:")
        if zeros:
            report.append(f"Warning! Using zeros for file content. Compressed filesystems will show optimistic results.")
        avg_time = sum(outResults[mode]) / len(outResults[mode])
        outResults[mode].sort()
        median_time = outResults[mode][len(outResults[mode]) // 2]
        one_percent_low_time = outResults[mode][int(len(outResults[mode]) * 0.01)]
        one_percent_high_time = outResults[mode][int(len(outResults[mode]) * 0.99)]
        zero_point_one_percent_high_time = outResults[mode][int(len(outResults[mode]) * 0.999)]
        highest_time = outResults[mode][-1]
        if 'write' in mode or 'read' in mode:
            total_size = file_size * file_count * process_count
            total_p_time = sum(outResults[mode]) / process_count
            total_bandwidth = total_size / total_p_time
            report.append(f"Total {mode} size:     \t{format_bytes(total_size)}B")
            report.append(f"Total {mode} speed:    \t{format_bytes(total_bandwidth)}B/s")
            report.append(f"                       \t{format_bytes(total_bandwidth * 8,use_1024_bytes=False)}b/s")
            report.append(f"With generation time:  \t{format_bytes(total_size / totalTime[mode])}B/s")
            report.append(f"                       \t{format_bytes(total_size / totalTime[mode] * 8,use_1024_bytes=False)}b/s")
            report.append(f"IO time %:             \t{total_p_time / totalTime[mode] * 100:.2f}%")
        report.append(f"Average {mode} time:   \t{avg_time:.4f} s")
        report.append(f"Median {mode} time:    \t{median_time:.4f} s")
        report.append(f"1 % low {mode} time:   \t{one_percent_low_time:.4f} s")
        report.append(f"1 % high {mode} time:  \t{one_percent_high_time:.4f} s")
        report.append(f"0.1 % high {mode} time:\t{zero_point_one_percent_high_time:.4f} s")
        report.append(f"Highest {mode} time:   \t{highest_time:.4f} s")
        report.append(f"-"*80)
        if threshold_to_report_anomaly and threshold_to_report_anomaly > 0 and one_percent_high_time > (one_percent_low_time * threshold_to_report_anomaly):
            report.append(f"Warning | iotest: 1% high is too high compared to 1% low!")
            report.append(f"Warning | iotest: 1% high is {one_percent_high_time:.4f} s, 1% low is {one_percent_low_time:.4f} s, threshold is {threshold_to_report_anomaly}")
        if message_end_point_address:
            
            payload = json.dumps({'content':f"Short Report for {mode} mode: Speed {format_bytes(total_bandwidth)}B/s @ {total_p_time / totalTime[mode] * 100:.2f}% IO time"})

            # use curl to send the message
            subprocess.run(['curl','-s','-X','POST','-H','Content-Type: application/json','-d',payload,message_end_point_address])
    

    tl.teeprint('\n'.join(report))

    # write the report to a file
    if not no_report and outResults:
        report_file_name = os.path.join(directory, f"iotest_{'-'.join(modes)}_{dir_str_repr}_fs={file_size}_fc={file_count}_pc={process_count}_{tl.currentDateTime}_report.txt")
        with open(report_file_name, "w") as f:
            f.write('\n'.join(report))
        tl.teeprint(f"Report written to {report_file_name}")
        print(f"Log file: {tl.logFileName}")

    return outResults

def climain():
    parser = argparse.ArgumentParser(description="Test total disk bandwidth. Default to comprehensive mode: write -> move -> stat -> read")
    parser.add_argument("-fs","--file_size", type=str, help="File size (default:30),  defaults to mb, can specify in t(b),g(b),m(b),k(b),b", default='30')
    parser.add_argument("-fc","--file_count", type=int, help="Number of files to create and read per process (default:50)",default=50)
    parser.add_argument("-t",'-pc',"--process_count", type=int, help="Number of processes to run concurrently (default:27)", default=36)
    parser.add_argument('-d',"--directory", type=str, help="Directory to put the files in (default:<pwd>)", default=os.getcwd())
    parser.add_argument('-ld',"--log_directory", type=str, help="Directory to put the log files in (default:/var/log/)", default='/var/log/')
    parser.add_argument("mode", nargs='*', type=str, help="""The mode the script will operate in (default:comprehensive).
 COMPREHENSIVE: async fully cached per thread write - index - read operation.
 WRITE: batched all thread write.
 READ: batched all thread read.
 INDEX: creates --file_count amount of index folders, stat it, then delete it.
 RWI: Execute Write - Index - Read mode sequentially in batch mode.""",choices=['comprehensive','read', 'write','random','index','benchmark','r','w','rw','wr','i','rwi','wri','c','b'], default="c")
    parser.add_argument("-q","--quiet", action="store_true", help="Suppress output, default True in new version",default=True)
    parser.add_argument("-v","--verbose", action="store_true", help="Verbose output",default=False)
    parser.add_argument("-S",'--stealth', action="store_true", help="Suppress verbose output and verbose log file",default=False)
    parser.add_argument('-nl',"--no_log", action="store_true", help="Do not write log files",default=False)
    parser.add_argument('-nr',"--no_report", action="store_true", help="Do not write report files",default=False)
    parser.add_argument("-z","--zeros", action="store_true", help="Use zeros instead of random numbers. Use this if you are sure no write compression is available. Potentially higher write accuracy.",default=False)
    parser.add_argument('-addr',"--message_end_point_address", type=str, help="The end point address of the message")
    parser.add_argument('--threshold_to_report_anomaly', type=int, help="The threshold to report if 1 percent high is higher then 1 percent low * <threshold_to_report_anomaly>",default=0)
    parser.add_argument("-V","--version", action="version", version=f"%(prog)s {version} with teeLogger {Tee_Logger.version} by pan@zopyr.us")
    args = parser.parse_args()

    # if we are on windows, set the log directory to the current directory
    if os.name == 'nt':
        args.log_directory = os.getcwd()

    if args.verbose:
        args.quiet = False
    if args.stealth:
        args.quiet = True
        args.no_report = True
    if args.no_log:
        args.log_directory = '/dev/null'
    #tl = Tee_Logger.teeLogger(args.log_directory,'iotest',2,10)
    tl = Tee_Logger.teeLogger(systemLogFileDir=args.log_directory,programName='iotest',compressLogAfterMonths=1,deleteLogAfterYears=3,suppressPrintout=args.stealth,noLog=args.no_log)
    
    tl.info(f'Arguments: {vars(args)}')

    modes = []
    for mode in args.mode:
        if mode == 'r':
            modes.append('read')
        elif mode == 'w':
            modes.append('write')
        elif mode == 'rw' or mode == 'wr':
            modes.append('write')
            modes.append('read')
        elif mode == 'i':
            modes.append('index')
        elif mode == 'rwi' or mode == 'wri':
            modes.append('write')
            modes.append('read')
            modes.append('index')
        elif mode == 'c':
            modes.append('comprehensive')
        elif mode == 'b':
            modes.append('benchmark')
    
        if 'write' in mode:
            modes.append('write')
        if 'read' in mode:
            modes.append('read')
        if 'random' in mode:
            modes.append('random')
        if 'index' in mode:
            modes.append('index')
        if 'comprehensive' in mode:
            modes.append('comprehensive')
        if 'benchmark' in mode:
            modes.append('benchmark')

    args.file_size = args.file_size.lower()
    if 'm' in args.file_size.lower():
        args.file_size = int(float(args.file_size.partition('m')[0]) * 1024 * 1024)
    elif 'g' in args.file_size.lower():
        args.file_size = int(float(args.file_size.partition('g')[0]) * 1024 * 1024 * 1024)
    elif 'k' in args.file_size.lower():
        args.file_size = int(float(args.file_size.partition('k')[0]) * 1024)
    elif 't' in args.file_size.lower():
        args.file_size = int(float(args.file_size.partition('t')[0]) * 1024 * 1024 * 1024 * 1024)
    elif 'b' in args.file_size.lower() and args.file_size.lower().partition('b')[0].isdigit():
        args.file_size = int(float(args.file_size.partition('b')[0]))
    else:
        args.file_size = int(float(args.file_size) * 1024 * 1024)


    # if no directory is provided, use the current directory
    if args.directory is None:
        args.directory = os.getcwd()

    if args.stealth:
        tl.info(f'Running in {modes} modes...')
        tl.info(f'File size: {format_bytes(args.file_size)}B')
        tl.info(f'Number of files: {args.file_count}')
        tl.info(f'Number of processes: {args.process_count}')
        if 'benchmark' not in modes and len(modes) == 1:
            tl.info(f'Writing to {args.directory}')
    else:
        tl.teeprint(f'Running in {modes} modes...')
        tl.teeprint(f'File size: {format_bytes(args.file_size)}B')
        tl.teeprint(f'Number of files: {args.file_count}')
        tl.teeprint(f'Number of processes: {args.process_count}')
        if 'benchmark' not in modes and len(modes) == 1:
            tl.teeprint(f'Writing to {args.directory}')
    main(args.file_size, args.file_count, args.process_count, args.directory,modes,args.quiet,args.zeros,tl=tl,stealth=args.stealth,message_end_point_address=args.message_end_point_address,no_report=args.no_report,threshold_to_report_anomaly=args.threshold_to_report_anomaly)
                
if __name__ == "__main__":
    climain()