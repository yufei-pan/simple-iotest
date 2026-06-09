#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Manager, Process

numpy_available = False
try:
	from numpy.random import SFC64, Generator # type: ignore
	numpy_available = True
except ImportError:
	numpy_available = False

version = '3.66.0'
COMMIT_DATE = '2026-06-08'
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

def worker_log(msg, level='info', quiet=False):
	if quiet:
		return
	printWithColor(msg, level)

# ------------------------------
try:
	import Tee_Logger # type: ignore
	assert float(Tee_Logger.version) >= 6.34
except Exception:
	class Tee_Logger:
		version = '0.1 inline'
		class teeLogger:
			def __init__(self, systemLogFileDir='.', programName=None, compressLogAfterMonths=2, 
				 deleteLogAfterYears=2, suppressPrintout=..., fileDescriptorLength=15,
				 noLog=False,callerStackDepth=-2,disable_colors=False, encoding = None,
				 in_place_compression = None, collapse_single_day_logs = ...,compression_level=...,
				 binary_mode = True):
				self.name = programName
				self.currentDateTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
				self.noLog = True
				if not noLog:
					print('Using inline logger, forcing noLog...')
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
def format_bytes(size, use_1024_bytes=None, to_int=False, to_str=False,str_format='.2f'):
	"""
	Format the size in bytes to a human-readable format or vice versa.
	From hpcp: https://github.com/yufei-pan/hpcp

	Args:
		size (int or str): The size in bytes or a string representation of the size.
		use_1024_bytes (bool, optional): Whether to use 1024 bytes as the base for conversion. If None, it will be determined automatically. Default is None.
		to_int (bool, optional): Whether to convert the size to an integer. Default is False.
		to_str (bool, optional): Whether to convert the size to a string representation. Default is False.
		str_format (str, optional): The format string to use when converting the size to a string. Default is '.2f'.

	Returns:
		int or str: The formatted size based on the provided arguments.

	Examples:
		>>> format_bytes(1500, use_1024_bytes=False)
		'1.50 K'
		>>> format_bytes('1.5 GiB', to_int=True)
		1610612736
		>>> format_bytes('1.5 GiB', to_str=True)
		'1.50 Gi'
		>>> format_bytes(1610612736, use_1024_bytes=True, to_str=True)
		'1.50 Gi'
		>>> format_bytes(1610612736, use_1024_bytes=False, to_str=True)
		'1.61 G'
	"""
	if to_int or isinstance(size, str):
		if isinstance(size, int):
			return size
		elif isinstance(size, str):
			# Use regular expression to split the numeric part from the unit, handling optional whitespace
			match = re.match(r"(\d+(\.\d+)?)\s*([a-zA-Z]*)", size)
			if not match:
				if to_str:
					return size
				print("Invalid size format. Expected format: 'number [unit]', e.g., '1.5 GiB' or '1.5GiB'")
				print(f"Got: {size}")
				return 0
			number, _, unit = match.groups()
			number = float(number)
			unit  = unit.strip().lower().rstrip('b')
			# Define the unit conversion dictionary
			if unit.endswith('i'):
				# this means we treat the unit as 1024 bytes if it ends with 'i'
				use_1024_bytes = True
			elif use_1024_bytes is None:
				use_1024_bytes = False
			unit  = unit.rstrip('i')
			if use_1024_bytes:
				power = 2**10
			else:
				power = 10**3
			unit_labels = {'': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5}
			if unit not in unit_labels:
				if to_str:
					return size
				print(f"Invalid unit '{unit}'. Expected one of {list(unit_labels.keys())}")
				return 0
			if to_str:
				return format_bytes(size=int(number * (power ** unit_labels[unit])), use_1024_bytes=use_1024_bytes, to_str=True, str_format=str_format)
			# Calculate the bytes
			return int(number * (power ** unit_labels[unit]))
		else:
			try:
				return int(size)
			except Exception:
				return 0
	elif to_str or isinstance(size, int) or isinstance(size, float):
		if isinstance(size, str):
			try:
				size = size.rstrip('B').rstrip('b')
				size = float(size.lower().strip())
			except Exception:
				return size
		# size is in bytes
		if use_1024_bytes or use_1024_bytes is None:
			power = 2**10
			n = 0
			power_labels = {0 : '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti', 5: 'Pi'}
			while size >= power:
				size /= power
				n += 1
			return f"{size:{str_format}}{' '}{power_labels[n]}"
		else:
			power = 10**3
			n = 0
			power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T', 5: 'P'}
			while size >= power:
				size /= power
				n += 1
			return f"{size:{str_format}}{' '}{power_labels[n]}"
	else:
		try:
			return format_bytes(float(size), use_1024_bytes)
		except Exception as e:
			import traceback
			print(f"Error: {e}")
			print(traceback.format_exc())
			print(f"Invalid size: {size}")
		return 0

def almost_urandom(n):
	try:
		if numpy_available:
			rng_sf = Generator(SFC64())
			n_u64 = (n + 7) // 8
			u64 = rng_sf.bit_generator.random_raw(n_u64)
			return u64.view('uint8')[:n].tobytes()
		else:
			return random.getrandbits(8 * n).to_bytes(n, 'big')
	except OverflowError:
		return almost_urandom(n // 2) + almost_urandom(n - n // 2)
	
def create_file(file_name, file_content, file_size, quiet=False):
	try:
		with open(file_name, "wb", buffering=0) as f:
			try:
				if os.name == 'posix':
					os.posix_fadvise(f.fileno(), 0, file_size, os.POSIX_FADV_DONTNEED)
			except: # noqa: E722
				worker_log('Failed to posix_fadvise', level='warning', quiet=quiet)
			start_write_time = time.perf_counter()
			try:
				if os.name == 'posix':
					os.writev(f.fileno(), [file_content])
				else:
					os.write(f.fileno(), file_content)
			except: # noqa: E722
				worker_log('Failed to write using os.writev, trying f.write', level='warning', quiet=quiet)
				f.write(file_content)
				f.flush()
			os.fsync(f.fileno())
			end_write_time = time.perf_counter()
			return start_write_time,end_write_time
	except Exception as e:
		import traceback
		worker_log(str(e), level='error', quiet=quiet)
		worker_log(traceback.format_exc(), level='error', quiet=quiet)
		return 0,time.perf_counter()

def move_file(src, dst, quiet=False):
	start_move_time = time.perf_counter()
	try:
		os.rename(src, dst)
		end_move_time = time.perf_counter()
		return start_move_time,end_move_time
	except Exception as e:
		import traceback
		worker_log(str(e), level='error', quiet=quiet)
		worker_log(traceback.format_exc(), level='error', quiet=quiet)
		return 0,time.perf_counter()

def read_file(file_name, file_content, file_size, quiet=False):
	b=bytearray(file_size)
	# check if file exists and size is correct
	try:
		if not os.path.isfile(file_name) or os.path.getsize(file_name) != file_size:
			# file does not exist or size is wrong, create it
			worker_log(f"File {file_name} does not exist or size is wrong, creating it...", level='error', quiet=quiet)
			create_file(file_name, file_content, file_size, quiet=quiet)

		with open(file_name, "rb", buffering=0) as f:
			try:
				if os.name == 'posix':
					os.posix_fadvise(f.fileno(), 0, file_size, os.POSIX_FADV_DONTNEED)
			except:  # noqa: E722
				worker_log('Failed to posix_fadvise', level='warning', quiet=quiet)
			start_read_time = time.perf_counter()
			try:
				if os.name == 'posix':
					os.readv(f.fileno(),[b])
				else:
					f.readinto(b)
			except:  # noqa: E722
				worker_log('Failed to read using os.readv, trying f.readinto', level='warning', quiet=quiet)
				f.readinto(b)
			end_read_time = time.perf_counter()
			return start_read_time,end_read_time
	except Exception as e:
		import traceback
		worker_log(str(e), level='error', quiet=quiet)
		worker_log(traceback.format_exc(), level='error', quiet=quiet)
		return 0,time.perf_counter()
	

def index_file(file_name, quiet=False):
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
		worker_log(str(e), level='error', quiet=quiet)
		worker_log(traceback.format_exc(), level='error', quiet=quiet)
		return 0,time.perf_counter()
	
def stat_file(file_name, quiet=False):
	start_stat_time = time.perf_counter()
	try:
		size = os.stat(file_name).st_size
		end_stat_time = time.perf_counter()
		return start_stat_time,end_stat_time,size
	except Exception as e:
		import traceback
		worker_log(str(e), level='error', quiet=quiet)
		worker_log(traceback.format_exc(), level='error', quiet=quiet)
		return 0,time.perf_counter(),0

def int_to_color(n, brightness_threshold=500):
	hash_value = hash(str(n))
	r = (hash_value >> 16) & 0xFF
	g = (hash_value >> 8) & 0xFF
	b = hash_value & 0xFF
	if (r + g + b) < brightness_threshold:
		return int_to_color(hash_value, brightness_threshold)
	return (r, g, b)

def worker(file_count, file_size, directory, results, mode, counter, quiet, zeros, thread_start_time):
	local_results = []
	r, g, b = int_to_color(os.getpid())
	worker_log(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} scheduled to start at {thread_start_time-time.perf_counter():.4f} later in {mode} mode.' + '\033[0m', quiet=quiet)
	if zeros:
		file_content = b'\x00' * file_size
	else:
		file_content = almost_urandom(file_size)
	if time.perf_counter() > thread_start_time:
		worker_log(f'Worker {counter} started late, expected start time {thread_start_time}, actual start time {time.perf_counter()}', level='error', quiet=quiet)
	worker_log(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} primed, waiting for {thread_start_time-time.perf_counter():.4f} seconds.' + '\033[0m', quiet=quiet)
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
			start_write_time,end_write_time = create_file(file_name, file_content, file_size, quiet=quiet)
			local_results.append(end_write_time- start_write_time)
			worker_log(f'\033[38;2;{r};{g};{b}m' + f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tWrote\tin {end_write_time-start_write_time} s"+ '\033[0m', quiet=quiet)
		elif mode == 'read':
			start_read_time,end_read_time = read_file(file_name, file_content, file_size, quiet)
			local_results.append(end_read_time- start_read_time)
			worker_log(f'\033[38;2;{r};{g};{b}m' +f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tRead\tin {end_read_time-start_read_time} s"+ '\033[0m', quiet=quiet)
		elif mode == 'random':
			# here we start a random read or write
			if bool(random.getrandbits(1)):
				start_write_time,end_write_time = create_file(file_name, file_content, file_size, quiet=quiet)
				local_results.append(end_write_time- start_write_time)
				worker_log(f'\033[38;2;{r};{g};{b}m' + f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tWrote\tin {end_write_time-start_write_time} s"+ '\033[0m', quiet=quiet)
			else:
				start_read_time,end_read_time = read_file(file_name, file_content, file_size, quiet)
				local_results.append(end_read_time- start_read_time)
				worker_log(f'\033[38;2;{r};{g};{b}m' +f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tRead\tin {end_read_time-start_read_time} s"+ '\033[0m', quiet=quiet)
		elif mode == 'comprehensive':
			resultList = []
			moved_name = file_name + ".moved"
			start_write_time,end_write_time = create_file(file_name, file_content, file_size, quiet=quiet)
			resultList.append(end_write_time- start_write_time)
			start_move_time,end_move_time = move_file(file_name, moved_name, quiet=quiet)
			resultList.append(end_move_time- start_move_time)
			start_stat_time,end_stat_time,size = stat_file(moved_name, quiet=quiet)
			if size != file_size:
				worker_log(f'{bcolors.critical}File {file_name} size is wrong, expected {file_size}, got {size}\033[0m', level='error', quiet=quiet)
			resultList.append(end_stat_time- start_stat_time)
			start_read_time,end_read_time = read_file(moved_name, file_content, file_size, quiet)
			resultList.append(end_read_time- start_read_time)
			try:
				os.remove(moved_name)
			except OSError:
				pass
			worker_log(f'\033[38;2;{r};{g};{b}m[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tW: {resultList[0]:.4f} s M: {resultList[1]:.4f} s S: {resultList[2]:.4f} s R: {resultList[3]:.4f} s\033[0m', quiet=quiet)
			local_results.append(resultList)
		elif mode == 'index':
			# This will test the indexing performance of the filesystem
			start_index_time,end_index_time = index_file(file_name, quiet)
			local_results.append(end_index_time- start_index_time)
			worker_log(f'\033[38;2;{r};{g};{b}m' +f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tIndexed\tin {end_index_time-start_index_time} s"+ '\033[0m', quiet=quiet)
	worker_log(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} finished.' + '\033[0m', quiet=quiet)
	results.extend(local_results)

def simultaneous_worker_dir(base_directory, run_id, counter):
	return os.path.join(base_directory, f".iotest_simul_{run_id}", f"worker_{counter}")

def layout_read_worker(file_count, file_size, base_directory, run_id, counter, quiet, zeros):
	if zeros:
		file_content = b'\x00' * file_size
	else:
		file_content = almost_urandom(file_size)
	worker_dir = simultaneous_worker_dir(base_directory, run_id, counter)
	os.makedirs(worker_dir, exist_ok=True)
	for i in range(file_count):
		i_str = str(i + 1).zfill(len(str(file_count)))
		file_name = os.path.join(worker_dir, f"read_{i_str}.bin")
		create_file(file_name, file_content, file_size, quiet=quiet)

def simultaneous_role_worker(role, file_count, file_size, base_directory, run_id, counter, results, quiet, zeros, thread_start_time):
	local_results = []
	r, g, b = int_to_color(os.getpid())
	worker_log(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} ({role}) scheduled to start at {thread_start_time-time.perf_counter():.4f} later.' + '\033[0m', quiet=quiet)
	if zeros:
		file_content = b'\x00' * file_size
	else:
		file_content = almost_urandom(file_size)
	while time.perf_counter() < thread_start_time:
		time.sleep(0.001)
	worker_dir = simultaneous_worker_dir(base_directory, run_id, counter)
	os.makedirs(worker_dir, exist_ok=True)
	file_prefix = 'write_' if role == 'write' else 'read_'
	for i in range(file_count):
		i_str = str(i + 1).zfill(len(str(file_count)))
		file_name = os.path.join(worker_dir, f"{file_prefix}{i_str}.bin")
		if role == 'write':
			start_time, end_time = create_file(file_name, file_content, file_size, quiet=quiet)
			local_results.append(end_time - start_time)
			worker_log(f'\033[38;2;{r};{g};{b}m' + f"[Process {os.getpid()}]\tFile {i_str}/{file_count}:\t{file_name}\tWrote\tin {end_time-start_time} s" + '\033[0m', quiet=quiet)
		else:
			start_time, end_time = read_file(file_name, file_content, file_size, quiet)
			local_results.append(end_time - start_time)
			worker_log(f'\033[38;2;{r};{g};{b}m' + f"[Process {os.getpid()}]\tFile {i_str}/{file_count}:\t{file_name}\tRead\tin {end_time-start_time} s" + '\033[0m', quiet=quiet)
	worker_log(f'\033[38;2;{r};{g};{b}m' + f'Worker {counter} ({role}) finished.' + '\033[0m', quiet=quiet)
	results.extend(local_results)

def cleanup_simultaneous_tree(directories, run_id, process_count):
	seen = set()
	for counter in range(process_count):
		base = directories[counter % len(directories)]
		simul_root = os.path.join(base, f".iotest_simul_{run_id}")
		if simul_root not in seen:
			seen.add(simul_root)
			if os.path.isdir(simul_root):
				shutil.rmtree(simul_root, ignore_errors=True)

def run_simultaneous_layout(file_count, file_size, directories, run_id, layout_worker_count,
		counter_padding_process_count, quiet, zeros, tl):
	tl.teeprint(f"Simultaneous mode: layout pass writing read files ({layout_worker_count} workers)...")
	layout_start = time.perf_counter()
	layout_processes = []
	for counter in range(layout_worker_count):
		assigned_directory = directories[counter % len(directories)]
		counter_str = str(counter).zfill(len(str(counter_padding_process_count)))
		p = Process(target=layout_read_worker, args=(file_count, file_size, assigned_directory, run_id, counter_str, quiet, zeros))
		layout_processes.append(p)
	for p in layout_processes:
		p.start()
	for p in layout_processes:
		p.join()
	tl.teeprint(f"Layout pass complete in {time.perf_counter() - layout_start:.4f} s")

def normalize_directories(directory):
	if isinstance(directory, str):
		return [directory]
	if directory is None:
		return [os.getcwd()]
	return list(directory)


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

def benchmarkGenSpeed(file_size, file_count,zeros,results,timeout=5):
	start_time = time.perf_counter()
	c = 0
	while True:
		if zeros:
			_ = b'\x00' * file_size
		else:
			_ = almost_urandom(file_size) 
		c += 1
		if (file_count > 0 and c >= file_count) or time.perf_counter() - start_time > timeout:
			break
	genTime = time.perf_counter() - start_time
	genSize = file_size * c
	genSpeed = genSize / genTime 
	results.append(genSpeed)

TUNING_TOLERANCE = 1.10
MAX_TUNING_ATTEMPTS = 20
SIMULTANEOUS_TUNING_BENCHMARK_TIME = 1

def supports_write_throughput_tuning(modes):
	return 'write' in modes or 'comprehensive' in modes or 'simultaneous' in modes

def supports_read_throughput_tuning(modes):
	return 'read' in modes or 'comprehensive' in modes or 'simultaneous' in modes

def throughput_status(measured, target):
	if measured is None:
		return 'missing'
	if measured < target:
		return 'below'
	if measured <= target * TUNING_TOLERANCE:
		return 'ok'
	return 'above'

def sync_bandwidth_bps(outResults, totalTime, file_size, file_count, process_count, mode, mode_process_counts=None):
	if mode not in outResults or mode not in totalTime or totalTime[mode] <= 0:
		return None
	effective_pc = (mode_process_counts or {}).get(mode, process_count)
	total_size = file_size * file_count * effective_pc
	return total_size / totalTime[mode]

def measure_role_bandwidth(result, role):
	return sync_bandwidth_bps(
		result['outResults'], result['totalTime'],
		result['file_size'], result['file_count'],
		result['process_count'], role,
		result.get('mode_process_counts'))

def main(file_size, file_count, process_count, directory,modes,quiet,zeros,tl=None,stealth=False,message_end_point_address=None,no_report=False,threshold_to_report_anomaly = 0,benchmarkTime = 5,tuning_attempt=False,simultaneous_write_count=None,simultaneous_run_id=None,simultaneous_skip_layout=False,simultaneous_skip_cleanup=False,simultaneous_layout_workers=None,simultaneous_layout_process_count=None):
	if tuning_attempt:
		no_report = True
		message_end_point_address = None
	if stealth:
		quiet = True
		no_report = True
	if not tl:
		tl = Tee_Logger.teeLogger(suppressPrintout=quiet)
	if isinstance(directory, str):
		directories = [directory]
	elif directory is None:
		directories = [os.getcwd()]
	else:
		directories = list(directory)
	if not directories:
		raise ValueError("At least one output directory must be provided")
	for output_dir in directories:
		os.makedirs(output_dir, exist_ok=True)
	primary_directory = directories[0]
	processes = []
	file_size = int(file_size)
	outResults = dict()
	totalTime = {}
	mode_process_counts = {}
	estimatedTotalMemory = file_size * process_count * 1.2
	phyFreeMemory = -1
	swapMemory = -1
	if modes == ['benchmark']:
		file_count = 0
	if os.path.exists('/proc/meminfo'):
		try:
			with open('/proc/meminfo') as f:
				for line in f:
					if 'MemAvailable' in line:
						phyFreeMemory = int(line.split()[1]) * 1024
					if 'SwapFree' in line:
						swapMemory = int(line.split()[1]) * 1024
		except Exception:
			tl.teeerror("Failed to read /proc/meminfo")
	# warn if the estimated total memory is more than 90% of the total free memory
	if phyFreeMemory + swapMemory > 0 and estimatedTotalMemory > phyFreeMemory + swapMemory:
		tl.teeerror("Estimated total memory usage is more than the all available memory to use.")
		tl.teeerror(f"Physical available memory: {format_bytes(phyFreeMemory)}B")
		tl.teeerror(f"Swap available memory: {format_bytes(swapMemory)}B")
		tl.teeerror(f"Estimated total memory usage: {format_bytes(estimatedTotalMemory)}B")
		process_count = max(int((phyFreeMemory + swapMemory) // file_size // 1.2), 1)
		tl.teeerror(f"Reducing the number of processes to {process_count}")
	estimatedTotalMemory = file_size * process_count * 1.2
	if phyFreeMemory > 0 and estimatedTotalMemory > phyFreeMemory * 0.9:
		tl.teeerror("Estimated total memory usage is more than 90% of the total non swap available memory.")
		tl.teeerror("You may want to reduce the file size or the number of processes.")
		tl.teeerror(f"Available memory: {format_bytes(phyFreeMemory)}B")
		tl.teeerror(f"Estimated total memory usage: {format_bytes(estimatedTotalMemory)}B")
		tl.teelog("If continuing, iotest will likely put very heavy pressure on swap memory and may lead to system crash.",level='critical')
		tl.teelog("Exit now (press Ctrl+C) or iotest will continue anyway in 15 seconds...",level='critical')
		time.sleep(16)
		tl.teelog("Warning! Continuing anyway... ",level='warning')
	with Manager() as manager:
		# bench mark file generation performance first
		results = manager.list()
		is_simultaneous = 'simultaneous' in modes
		if is_simultaneous:
			if simultaneous_write_count is not None:
				gen_benchmark_workers = simultaneous_write_count
			else:
				gen_benchmark_workers = process_count // 2
		else:
			gen_benchmark_workers = process_count
		effective_benchmark_time = benchmarkTime
		if is_simultaneous and tuning_attempt:
			effective_benchmark_time = SIMULTANEOUS_TUNING_BENCHMARK_TIME
		if is_simultaneous and gen_benchmark_workers == 0:
			genSpeed = 0
			genTimeCalc = 0
			processStartDelay = 0.005 * process_count + 1
			tl.teeprint("Skipping file generation benchmark (no write workers in simultaneous mode)")
		else:
			if zeros:
				generator = 'zeros'
			elif numpy_available:
				generator = 'numpy SFC64'
			else:
				generator = 'random.getrandbits'
			tl.teeprint(f"Benchmarking file generation performance... using {gen_benchmark_workers}x {generator} with {format_bytes(file_size)}B files * {file_count if file_count else '∞'} for {effective_benchmark_time}s")
			for counter in range(gen_benchmark_workers):
				p = Process(target=benchmarkGenSpeed, args=(file_size, file_count,zeros,results,effective_benchmark_time))
				processes.append(p)
			genStartTime = time.perf_counter()
			for p in processes:
				p.start()
			for p in processes:
				p.join()
			processes.clear()
			genTime = time.perf_counter() - genStartTime
			genSpeed = sum(results)
			if genSpeed > 0:
				genTimeCalc = file_size * gen_benchmark_workers / genSpeed
				processStartDelay = 0.005 * process_count + 1 + genTimeCalc
			else:
				genTimeCalc = 0
				processStartDelay = 0.005 * process_count + 1
				tl.teeerror("File generation benchmark returned zero speed; using minimal process start delay")
			tl.teeprint(f"Generation speed:      \t{format_bytes(genSpeed)}B/s")
			tl.teeprint(f"                       \t{format_bytes(genSpeed * 8,use_1024_bytes=False)}b/s")
			tl.teeprint(f"Generation test time:  \t{genTime:.4f} s")
			tl.teeprint(f"Gen time calculated:   \t{genTimeCalc:.4f} s")
			tl.teeprint(f"Process start delay:   \t{processStartDelay:.4f} s")
		for mode in modes:
			if mode == 'benchmark':
				continue
			if mode == 'simultaneous':
				if process_count < 2:
					raise ValueError("simultaneous mode requires process_count >= 2")
				if simultaneous_write_count is not None:
					write_count = simultaneous_write_count
					read_count = process_count - write_count
				else:
					write_count = process_count // 2
					read_count = process_count - write_count
				if write_count < 1 or read_count < 1:
					raise ValueError("simultaneous mode requires at least 1 writer and 1 reader")
				mode_process_counts = {'write': write_count, 'read': read_count}
				run_id = simultaneous_run_id or tl.currentDateTime
				layout_padding_pc = simultaneous_layout_process_count or process_count
				layout_worker_count = simultaneous_layout_workers if simultaneous_layout_workers is not None else process_count
				try:
					if not simultaneous_skip_layout:
						run_simultaneous_layout(file_count, file_size, directories, run_id, layout_worker_count,
							layout_padding_pc, quiet, zeros, tl)
					write_results = manager.list()
					read_results = manager.list()
					write_processes = []
					read_processes = []
					thread_start_time = time.perf_counter() + processStartDelay
					tl.teeprint(f"Simultaneous mode: {write_count} writers + {read_count} readers...")
					for counter in range(write_count):
						assigned_directory = directories[counter % len(directories)]
						counter_str = str(counter).zfill(len(str(process_count)))
						p = Process(target=simultaneous_role_worker, args=('write', file_count, file_size, assigned_directory, run_id, counter_str, write_results, quiet, zeros, thread_start_time))
						write_processes.append(p)
					for counter in range(read_count):
						assigned_directory = directories[counter % len(directories)]
						counter_str = str(counter).zfill(len(str(process_count)))
						p = Process(target=simultaneous_role_worker, args=('read', file_count, file_size, assigned_directory, run_id, counter_str, read_results, quiet, zeros, thread_start_time))
						read_processes.append(p)
					totalStartTime = thread_start_time
					for p in write_processes + read_processes:
						p.start()
					for p in write_processes:
						p.join()
					write_sync_time = time.perf_counter() - totalStartTime
					for p in read_processes:
						p.join()
					read_sync_time = time.perf_counter() - totalStartTime
					addToDicWithoutOverwrite(outResults, 'write', list(write_results))
					addToDicWithoutOverwrite(outResults, 'read', list(read_results))
					addToDicWithoutOverwrite(totalTime, 'write', write_sync_time)
					addToDicWithoutOverwrite(totalTime, 'read', read_sync_time)
				finally:
					if not simultaneous_skip_cleanup:
						cleanup_simultaneous_tree(directories, run_id, layout_padding_pc)
						tl.teeprint("Simultaneous mode: cleaned up temporary files.")
				continue
			processes = []
			results = manager.list()
			thread_start_time = time.perf_counter() + processStartDelay
			for counter in range(process_count):
				assigned_directory = directories[counter % len(directories)]
				counter = str(counter).zfill(len(str(process_count)))
				p = Process(target=worker, args=(file_count, file_size, assigned_directory, results, mode, counter, quiet, zeros, thread_start_time))
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
				if sumAllTime > 0:
					addToDicWithoutOverwrite(totalTime,'write',totalEndTime * (sumWTime / sumAllTime))
					addToDicWithoutOverwrite(totalTime,'move',totalEndTime * (sumMTime / sumAllTime))
					addToDicWithoutOverwrite(totalTime,'stat',totalEndTime * (sumSTime / sumAllTime))
					addToDicWithoutOverwrite(totalTime,'read',totalEndTime * (sumRTime / sumAllTime))
				else:
					addToDicWithoutOverwrite(totalTime,'write',totalEndTime)
					addToDicWithoutOverwrite(totalTime,'move',totalEndTime)
					addToDicWithoutOverwrite(totalTime,'stat',totalEndTime)
					addToDicWithoutOverwrite(totalTime,'read',totalEndTime)
			else:
				addToDicWithoutOverwrite(outResults,mode,list(results))
				addToDicWithoutOverwrite(totalTime,mode,totalEndTime)
	# write the outResults to a csv file
	if not no_report and outResults:
		dir_str_repr = "_".join([d.replace('/','-').replace('\\','-').replace(':','-').replace('--','-').replace(' ','_') for d in directories])
		csv_file_name = os.path.join(primary_directory, f"iotest_{'-'.join(modes)}_{dir_str_repr}_fs={file_size}_fc={file_count}_pc={process_count}_{tl.currentDateTime}.csv")
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
		report.append("*"*80)
		report.append(f"Report for {mode} mode:")
		if zeros:
			report.append("Warning! Using zeros for file content. Compressed filesystems will show optimistic results.")
		avg_time = sum(outResults[mode]) / len(outResults[mode])
		outResults[mode].sort()
		median_time = outResults[mode][len(outResults[mode]) // 2]
		one_percent_low_time = outResults[mode][int(len(outResults[mode]) * 0.01)]
		one_percent_high_time = outResults[mode][int(len(outResults[mode]) * 0.99)]
		zero_point_one_percent_high_time = outResults[mode][int(len(outResults[mode]) * 0.999)]
		highest_time = outResults[mode][-1]
		if 'write' in mode or 'read' in mode:
			effective_pc = mode_process_counts.get(mode, process_count)
			total_size = file_size * file_count * effective_pc
			total_p_time = sum(outResults[mode]) / effective_pc
			total_bandwidth = total_size / total_p_time
			report.append(f"Total {mode} size:      \t{format_bytes(total_size)}B")
			report.append(f"{mode} bandwidth (call):\t{format_bytes(total_bandwidth)}B/s")
			report.append(f"                       \t{format_bytes(total_bandwidth * 8,use_1024_bytes=False)}bps")
			report.append(f"{mode} bandwidth (sync):\t{format_bytes(total_size / totalTime[mode])}B/s")
			report.append(f"                       \t{format_bytes(total_size / totalTime[mode] * 8,use_1024_bytes=False)}bps")
			report.append(f"IO time % (IO wait):   \t{total_p_time / totalTime[mode] * 100:.2f}%")
		report.append(f"Average {mode} time:    \t{avg_time:.4f} s")
		report.append(f"Median {mode} time:     \t{median_time:.4f} s")
		report.append(f"1 % low {mode} time:    \t{one_percent_low_time:.4f} s")
		report.append(f"1 % high {mode} time:   \t{one_percent_high_time:.4f} s")
		report.append(f"0.1 % high {mode} time: \t{zero_point_one_percent_high_time:.4f} s")
		report.append(f"Highest {mode} time:    \t{highest_time:.4f} s")
		report.append("-"*80)
		if threshold_to_report_anomaly and threshold_to_report_anomaly > 0 and one_percent_high_time > (one_percent_low_time * threshold_to_report_anomaly):
			report.append("Warning | iotest: 1% high is too high compared to 1% low!")
			report.append(f"Warning | iotest: 1% high is {one_percent_high_time:.4f} s, 1% low is {one_percent_low_time:.4f} s, threshold is {threshold_to_report_anomaly}")
		if message_end_point_address:
			if 'write' in mode or 'read' in mode:
				payload = json.dumps({'content': f"Short Report for {mode} mode: Speed {format_bytes(total_bandwidth)}B/s @ {total_p_time / totalTime[mode] * 100:.2f}% IO time"})
			else:
				payload = json.dumps({'content': f"Short Report for {mode} mode: avg {avg_time:.4f} s, median {median_time:.4f} s"})
			subprocess.run(['curl', '-s', '-X', 'POST', '-H', 'Content-Type: application/json', '-d', payload, message_end_point_address])
	tl.teeprint('\n'.join(report))
	# write the report to a file
	if not no_report and outResults:
		report_file_name = os.path.join(primary_directory, f"iotest_{'-'.join(modes)}_{dir_str_repr}_fs={file_size}_fc={file_count}_pc={process_count}_{tl.currentDateTime}_report.txt")
		with open(report_file_name, "w") as f:
			f.write('\n'.join(report))
		tl.teeprint(f"Report written to {report_file_name}")
		print(f"Log file: {tl.logFileName}")
	return {
		'outResults': outResults,
		'totalTime': totalTime,
		'process_count': process_count,
		'file_size': file_size,
		'file_count': file_count,
		'mode_process_counts': mode_process_counts,
	}

MODE_ALIASES = {
	'r': ['read'],
	'w': ['write'],
	'rw': ['write', 'read'],
	'wr': ['write', 'read'],
	'i': ['index'],
	'rwi': ['write', 'read', 'index'],
	'wri': ['write', 'read', 'index'],
	'c': ['comprehensive'],
	'b': ['benchmark'],
	'read': ['read'],
	'write': ['write'],
	'random': ['random'],
	'index': ['index'],
	'comprehensive': ['comprehensive'],
	'benchmark': ['benchmark'],
	's': ['simultaneous'],
	'simultaneous': ['simultaneous'],
}

def parse_modes(mode_args):
	if isinstance(mode_args, str):
		mode_args = [mode_args]
	modes = []
	for token in mode_args:
		key = token.lower()
		if key in MODE_ALIASES:
			modes.extend(MODE_ALIASES[key])
	return modes

def _simultaneous_session_kwargs(run_id, max_pc, skip_layout=True, skip_cleanup=True):
	return {
		'simultaneous_run_id': run_id,
		'simultaneous_skip_layout': skip_layout,
		'simultaneous_skip_cleanup': skip_cleanup,
		'simultaneous_layout_process_count': max_pc,
	}

def _run_tuning_main(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
		stealth, threshold_to_report_anomaly, benchmarkTime, simultaneous_write_count=None,
		simultaneous_session=None):
	kwargs = simultaneous_session or {}
	return main(file_size, file_count, process_count, directory, modes, quiet, zeros, tl=tl,
		stealth=stealth, message_end_point_address=None, no_report=True,
		threshold_to_report_anomaly=threshold_to_report_anomaly, benchmarkTime=benchmarkTime,
		tuning_attempt=True, simultaneous_write_count=simultaneous_write_count, **kwargs)

def _finalize_tuning_run(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
		stealth, message_end_point_address, no_report, threshold_to_report_anomaly, benchmarkTime,
		simultaneous_write_count=None, simultaneous_session=None):
	if not no_report or message_end_point_address:
		kwargs = simultaneous_session or {}
		main(file_size, file_count, process_count, directory, modes, quiet, zeros, tl=tl,
			stealth=stealth, message_end_point_address=message_end_point_address,
			no_report=no_report, threshold_to_report_anomaly=threshold_to_report_anomaly,
			benchmarkTime=benchmarkTime, tuning_attempt=False,
			simultaneous_write_count=simultaneous_write_count, **kwargs)

def tune_single_role(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
		role, target_bps, stealth=False, message_end_point_address=None, no_report=False,
		threshold_to_report_anomaly=0, benchmarkTime=5, simultaneous_write_count=None):
	pc = process_count
	tl.teeprint(f"Target throughput tuning: {role} target={format_bytes(target_bps)}B/s, starting process_count={pc}")
	for attempt in range(1, MAX_TUNING_ATTEMPTS + 1):
		result = _run_tuning_main(file_size, file_count, pc, directory, modes, quiet, zeros, tl,
			stealth, threshold_to_report_anomaly, benchmarkTime, simultaneous_write_count)
		effective_pc = result['process_count']
		measured = measure_role_bandwidth(result, role)
		if measured is None:
			tl.teeerror(f"Cannot measure sync bandwidth for {role}")
			sys.exit(1)
		status = throughput_status(measured, target_bps)
		tl.teeprint(f"tpt attempt {attempt}: pc={effective_pc} {role}={format_bytes(measured)}B/s "
			f"target={format_bytes(target_bps)}B/s (max {format_bytes(target_bps * TUNING_TOLERANCE)}B/s)")
		if status == 'below':
			if attempt == 1:
				tl.teeerror(f"Cannot meet {role} target throughput {format_bytes(target_bps)}B/s: "
					f"measured {format_bytes(measured)}B/s at process_count={effective_pc}. "
					f"Not increasing process_count.")
			else:
				tl.teeerror(f"Overshot {role} target while tuning: measured {format_bytes(measured)}B/s "
					f"at process_count={effective_pc}.")
			sys.exit(1)
		if status == 'ok':
			tl.teeprint(f"{role} target throughput reached: process_count={effective_pc}, "
				f"measured={format_bytes(measured)}B/s")
			_finalize_tuning_run(file_size, file_count, effective_pc, directory, modes, quiet, zeros, tl,
				stealth, message_end_point_address, no_report, threshold_to_report_anomaly,
				benchmarkTime, simultaneous_write_count)
			return 0
		if effective_pc <= 1:
			tl.teeprint(f"At process_count=1, {role} measured {format_bytes(measured)}B/s still above target; "
				f"accepting as best effort.")
			_finalize_tuning_run(file_size, file_count, 1, directory, modes, quiet, zeros, tl,
				stealth, message_end_point_address, no_report, threshold_to_report_anomaly,
				benchmarkTime, simultaneous_write_count)
			return 0
		new_pc = max(1, int(effective_pc * target_bps / measured))
		if new_pc >= effective_pc:
			new_pc = effective_pc - 1
		pc = new_pc
	tl.teeerror("Target throughput tuning exceeded maximum attempts.")
	sys.exit(1)

def tune_dual_role_shared_pc(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
		write_target, read_target, stealth=False, message_end_point_address=None, no_report=False,
		threshold_to_report_anomaly=0, benchmarkTime=5):
	pc = process_count
	tl.teeprint(f"Target throughput tuning: write target={format_bytes(write_target)}B/s, "
		f"read target={format_bytes(read_target)}B/s, starting process_count={pc}")
	for attempt in range(1, MAX_TUNING_ATTEMPTS + 1):
		result = _run_tuning_main(file_size, file_count, pc, directory, modes, quiet, zeros, tl,
			stealth, threshold_to_report_anomaly, benchmarkTime)
		effective_pc = result['process_count']
		write_measured = measure_role_bandwidth(result, 'write')
		read_measured = measure_role_bandwidth(result, 'read')
		if write_measured is None or read_measured is None:
			tl.teeerror("Cannot measure sync bandwidth for write/read")
			sys.exit(1)
		write_status = throughput_status(write_measured, write_target)
		read_status = throughput_status(read_measured, read_target)
		tl.teeprint(f"tpt attempt {attempt}: pc={effective_pc} write={format_bytes(write_measured)}B/s "
			f"(target {format_bytes(write_target)}B/s) read={format_bytes(read_measured)}B/s "
			f"(target {format_bytes(read_target)}B/s)")
		if write_status == 'below' or read_status == 'below':
			if attempt == 1:
				if write_status == 'below':
					tl.teeerror(f"Cannot meet write target throughput {format_bytes(write_target)}B/s: "
						f"measured {format_bytes(write_measured)}B/s at process_count={effective_pc}.")
				if read_status == 'below':
					tl.teeerror(f"Cannot meet read target throughput {format_bytes(read_target)}B/s: "
						f"measured {format_bytes(read_measured)}B/s at process_count={effective_pc}.")
				tl.teeerror("Not increasing process_count.")
			else:
				tl.teeerror("Overshot target while tuning shared process_count.")
			sys.exit(1)
		if write_status == 'ok' or read_status == 'ok':
			tl.teeprint(f"Target throughput reached: process_count={effective_pc}, "
				f"write={format_bytes(write_measured)}B/s, read={format_bytes(read_measured)}B/s")
			_finalize_tuning_run(file_size, file_count, effective_pc, directory, modes, quiet, zeros, tl,
				stealth, message_end_point_address, no_report, threshold_to_report_anomaly, benchmarkTime)
			return 0
		if effective_pc <= 1:
			tl.teeprint("At process_count=1, both write and read still above target; accepting as best effort.")
			_finalize_tuning_run(file_size, file_count, 1, directory, modes, quiet, zeros, tl,
				stealth, message_end_point_address, no_report, threshold_to_report_anomaly, benchmarkTime)
			return 0
		new_pc = max(1, min(int(effective_pc * write_target / write_measured),
			int(effective_pc * read_target / read_measured)))
		if new_pc >= effective_pc:
			new_pc = effective_pc - 1
		pc = new_pc
	tl.teeerror("Target throughput tuning exceeded maximum attempts.")
	sys.exit(1)

def estimate_simultaneous_split(process_count, write_target, read_target):
	total_target = write_target + read_target
	write_count = max(1, min(process_count - 1, int(round(process_count * write_target / total_target))))
	read_count = process_count - write_count
	return write_count, read_count

def tune_simultaneous(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
		write_target, read_target, stealth=False, message_end_point_address=None, no_report=False,
		threshold_to_report_anomaly=0, benchmarkTime=5):
	benchmarkTime = SIMULTANEOUS_TUNING_BENCHMARK_TIME
	max_pc = process_count
	pc = process_count
	directories = normalize_directories(directory)
	for output_dir in directories:
		os.makedirs(output_dir, exist_ok=True)
	run_id = tl.currentDateTime
	layout_workers = max(max_pc - 1, 1)
	sim_session = _simultaneous_session_kwargs(run_id, max_pc)
	if write_target and read_target:
		write_count, read_count = estimate_simultaneous_split(pc, write_target, read_target)
		split_note = (f"estimated {write_count} writers + {read_count} readers from target ratio "
			f"({format_bytes(write_target)}B/s write / {format_bytes(read_target)}B/s read)")
	else:
		write_count = pc // 2
		read_count = pc - write_count
		split_note = f"{write_count} writers + {read_count} readers"
	target_desc = []
	if write_target:
		target_desc.append(f"write={format_bytes(write_target)}B/s")
	if read_target:
		target_desc.append(f"read={format_bytes(read_target)}B/s")
	tl.teeprint(f"Simultaneous throughput tuning: {', '.join(target_desc)}, "
		f"starting process_count={pc} ({split_note})")
	def reset_split_for_pc(new_pc):
		nonlocal write_count, read_count
		if write_target and read_target:
			write_count, read_count = estimate_simultaneous_split(new_pc, write_target, read_target)
		else:
			write_count = new_pc // 2
			read_count = new_pc - write_count
	def finalize_simultaneous_tuning():
		_finalize_tuning_run(file_size, file_count, pc, directory, modes, quiet, zeros, tl,
			stealth, message_end_point_address, no_report, threshold_to_report_anomaly,
			benchmarkTime, write_count, simultaneous_session=sim_session)
	last_valid = None
	def note_valid_config():
		nonlocal last_valid
		last_valid = {
			'pc': pc,
			'write_count': write_count,
			'read_count': read_count,
			'write_measured': write_measured,
			'read_measured': read_measured,
		}
	def accept_last_valid(reason):
		nonlocal pc, write_count, read_count
		if last_valid is None:
			return False
		pc = last_valid['pc']
		write_count = last_valid['write_count']
		read_count = last_valid['read_count']
		tl.teeprint(f"{reason} Accepting {write_count}w+{read_count}r at pc={pc} "
			f"(write={format_bytes(last_valid['write_measured'])}B/s, "
			f"read={format_bytes(last_valid['read_measured'])}B/s).")
		finalize_simultaneous_tuning()
		return True
	try:
		run_simultaneous_layout(file_count, file_size, directories, run_id, layout_workers, max_pc, quiet, zeros, tl)
		for attempt in range(1, MAX_TUNING_ATTEMPTS + 1):
			result = _run_tuning_main(file_size, file_count, pc, directory, modes, quiet, zeros, tl,
				stealth, threshold_to_report_anomaly, benchmarkTime, write_count,
				simultaneous_session=sim_session)
			write_measured = measure_role_bandwidth(result, 'write')
			read_measured = measure_role_bandwidth(result, 'read')
			if write_measured is None or read_measured is None:
				tl.teeerror("Cannot measure sync bandwidth for simultaneous write/read")
				sys.exit(1)
			write_status = throughput_status(write_measured, write_target) if write_target else None
			read_status = throughput_status(read_measured, read_target) if read_target else None
			tl.teeprint(f"tpt attempt {attempt}: pc={pc} ({write_count}w+{read_count}r) "
				f"write={format_bytes(write_measured)}B/s read={format_bytes(read_measured)}B/s")
			if write_target and not read_target:
				if write_status != 'below':
					note_valid_config()
				if write_status == 'below':
					if accept_last_valid("Cannot reduce write workers without missing write target;"):
						return 0
					if read_count > 1:
						write_count += 1
						read_count -= 1
						continue
					tl.teeerror(f"Cannot meet write target throughput {format_bytes(write_target)}B/s: "
						f"measured {format_bytes(write_measured)}B/s.")
					sys.exit(1)
				if write_status == 'ok':
					tl.teeprint(f"Write target reached: {write_count} writers + {read_count} readers, "
						f"measured={format_bytes(write_measured)}B/s")
					finalize_simultaneous_tuning()
					return 0
				if write_count > 1:
					write_count -= 1
					read_count += 1
					continue
				if pc > 2:
					pc -= 1
					reset_split_for_pc(pc)
					continue
				tl.teeprint("At minimum simultaneous configuration; accepting as best effort.")
				finalize_simultaneous_tuning()
				return 0
			if read_target and not write_target:
				if read_status != 'below':
					note_valid_config()
				if read_status == 'below':
					if accept_last_valid("Cannot reduce read workers without missing read target;"):
						return 0
					if write_count > 1:
						write_count -= 1
						read_count += 1
						continue
					tl.teeerror(f"Cannot meet read target throughput {format_bytes(read_target)}B/s: "
						f"measured {format_bytes(read_measured)}B/s.")
					sys.exit(1)
				if read_status == 'ok':
					tl.teeprint(f"Read target reached: {write_count} writers + {read_count} readers, "
						f"measured={format_bytes(read_measured)}B/s")
					finalize_simultaneous_tuning()
					return 0
				if read_count > 1:
					write_count += 1
					read_count -= 1
					continue
				if pc > 2:
					pc -= 1
					reset_split_for_pc(pc)
					continue
				tl.teeprint("At minimum simultaneous configuration; accepting as best effort.")
				finalize_simultaneous_tuning()
				return 0
			if write_status == 'below' and read_status == 'below':
				tl.teeerror("Cannot meet both write and read target throughput at current process_count.")
				tl.teeerror(f"Write measured {format_bytes(write_measured)}B/s "
					f"(target {format_bytes(write_target)}B/s).")
				tl.teeerror(f"Read measured {format_bytes(read_measured)}B/s "
					f"(target {format_bytes(read_target)}B/s).")
				sys.exit(1)
			if write_status != 'below' and read_status != 'below':
				note_valid_config()
			if write_status == 'below' or read_status == 'below':
				if accept_last_valid("Cannot adjust worker ratio without missing a target;"):
					return 0
				if write_status == 'below' and read_count > 1:
					write_count += 1
					read_count -= 1
					continue
				if read_status == 'below' and write_count > 1:
					write_count -= 1
					read_count += 1
					continue
				tl.teeerror("Cannot balance simultaneous write/read ratio to meet both targets.")
				sys.exit(1)
			if write_status == 'ok' and read_status == 'ok':
				if pc <= 2:
					tl.teeprint(f"Both targets reached: pc={pc} ({write_count}w+{read_count}r), "
						f"write={format_bytes(write_measured)}B/s, read={format_bytes(read_measured)}B/s")
					finalize_simultaneous_tuning()
					return 0
				pc -= 1
				reset_split_for_pc(pc)
				continue
			if write_status == 'above' and read_status == 'above':
				write_ratio = write_measured / write_target
				read_ratio = read_measured / read_target
				if write_ratio > read_ratio and write_count > 1:
					write_count -= 1
					read_count += 1
					continue
				if read_ratio > write_ratio and read_count > 1:
					write_count += 1
					read_count -= 1
					continue
				if pc > 2:
					pc -= 1
					reset_split_for_pc(pc)
					continue
				tl.teeprint("At minimum simultaneous configuration with both above target; accepting as best effort.")
				finalize_simultaneous_tuning()
				return 0
			if write_status == 'above' and read_status == 'ok':
				if write_count > 1:
					write_count -= 1
					read_count += 1
					continue
				if pc > 2:
					pc -= 1
					reset_split_for_pc(pc)
					continue
				finalize_simultaneous_tuning()
				return 0
			if read_status == 'above' and write_status == 'ok':
				if read_count > 1:
					write_count += 1
					read_count -= 1
					continue
				if pc > 2:
					pc -= 1
					reset_split_for_pc(pc)
					continue
				finalize_simultaneous_tuning()
				return 0
		tl.teeerror("Simultaneous throughput tuning exceeded maximum attempts.")
		sys.exit(1)
	finally:
		cleanup_simultaneous_tree(directories, run_id, max_pc)
		tl.teeprint("Simultaneous mode: cleaned up temporary files.")

def tune_for_targets(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
		write_target, read_target, stealth=False, message_end_point_address=None, no_report=False,
		threshold_to_report_anomaly=0, benchmarkTime=5):
	if 'simultaneous' in modes:
		tune_simultaneous(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
			write_target, read_target, stealth=stealth, message_end_point_address=message_end_point_address,
			no_report=no_report, threshold_to_report_anomaly=threshold_to_report_anomaly,
			benchmarkTime=benchmarkTime)
		return
	is_comprehensive = 'comprehensive' in modes
	is_dual_phase = 'write' in modes and 'read' in modes and not is_comprehensive
	if write_target and read_target and (is_comprehensive or is_dual_phase):
		tune_dual_role_shared_pc(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
			write_target, read_target, stealth=stealth, message_end_point_address=message_end_point_address,
			no_report=no_report, threshold_to_report_anomaly=threshold_to_report_anomaly,
			benchmarkTime=benchmarkTime)
		return
	if write_target:
		tune_single_role(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
			'write', write_target, stealth=stealth, message_end_point_address=message_end_point_address,
			no_report=no_report, threshold_to_report_anomaly=threshold_to_report_anomaly,
			benchmarkTime=benchmarkTime)
		return
	if read_target:
		tune_single_role(file_size, file_count, process_count, directory, modes, quiet, zeros, tl,
			'read', read_target, stealth=stealth, message_end_point_address=message_end_point_address,
			no_report=no_report, threshold_to_report_anomaly=threshold_to_report_anomaly,
			benchmarkTime=benchmarkTime)

def climain():
	default_file_size_str = '30'
	default_file_count = 50
	default_process_count = 36

	def parse_size_arg(size_arg, unitless_as_mb=True):
		size_arg = str(size_arg).strip().lower()
		if 'm' in size_arg:
			return int(float(size_arg.partition('m')[0]) * 1024 * 1024)
		if 'g' in size_arg:
			return int(float(size_arg.partition('g')[0]) * 1024 * 1024 * 1024)
		if 'k' in size_arg:
			return int(float(size_arg.partition('k')[0]) * 1024)
		if 't' in size_arg:
			return int(float(size_arg.partition('t')[0]) * 1024 * 1024 * 1024 * 1024)
		if 'b' in size_arg and size_arg.partition('b')[0].replace('.', '', 1).isdigit():
			return int(float(size_arg.partition('b')[0]))
		if unitless_as_mb:
			return int(float(size_arg) * 1024 * 1024)
		return int(float(size_arg))

	parser = argparse.ArgumentParser(description="Test total disk bandwidth. Default mode is write (w). Comprehensive mode: write -> move -> stat -> read")
	parser.add_argument("-fs","--file_size", type=str, help="File size (default:30), defaults to mb, can specify in t(b),g(b),m(b),k(b),b", default=None)
	parser.add_argument("-fc","--file_count", type=int, help="Number of files to create and read per process (default:50)",default=None)
	parser.add_argument("-t",'-pc',"--process_count", type=int, help="Number of processes to run concurrently (default:36)", default=None)
	parser.add_argument("-ts","--total_size", type=str, help="Total size target. If -fs/-fc/-pc is omitted, auto-calculate one missing arg to fit this total size.")
	parser.add_argument('-d',"--directory", action='append', type=str, help="Directory to put the files in. Repeat -d to round-robin workers across directories (default:<pwd>)", default=None)
	parser.add_argument('-ld',"--log_directory", type=str, help="Directory to put the log files in (default:/var/log/)", default='/var/log/')
	parser.add_argument("mode", nargs='*', type=str, help="""The mode the script will operate in (default: write / w).
 COMPREHENSIVE: per-thread write -> move -> stat -> read (what your code sees).
 WRITE: batched all-thread write.
 READ: batched all-thread read.
 INDEX: creates --file_count index folders, stat each, then delete.
 RWI / WRI: run write, then read, then index sequentially in batch mode.
 SIMULTANEOUS / S: layout write pass for read files, then concurrent write+read on separate files; cleans up afterward.""",choices=['read', 'write','random','index','benchmark','comprehensive','simultaneous','r','w','rw','wr','i','rwi','wri','c','b','s'], default="w")
	parser.add_argument("-q","--quiet", action="store_true", help="Suppress output, default True in new version",default=True)
	parser.add_argument("-v","--verbose", action="store_true", help="Verbose output",default=False)
	parser.add_argument("-S",'--stealth', action="store_true", help="Suppress verbose output and verbose log file",default=False)
	parser.add_argument('-nl',"--no_log", action="store_true", help="Do not write log files",default=False)
	parser.add_argument('-nr',"--no_report", action="store_true", help="Do not write report files",default=False)
	parser.add_argument("-z","--zeros", action="store_true", help="Use zeros instead of random numbers. Use this if you are sure no write compression is available. Potentially higher write accuracy.",default=False)
	parser.add_argument('-addr',"--message_end_point_address", type=str, help="The end point address of the message")
	parser.add_argument('--threshold_to_report_anomaly', type=int, help="The threshold to report if 1 percent high is higher then 1 percent low * <threshold_to_report_anomaly>",default=0)
	parser.add_argument('-bt',"--benchmark_time", type=int, help="The time in seconds to run the benchmark for file generation speed",default=5)
	parser.add_argument("-wtpt", "--write_target_throughput", type=str,
		help="Target sync write throughput (e.g. 500m, 2g). Auto-tunes the first write pass.")
	parser.add_argument("-rtpt", "--read_target_throughput", type=str,
		help="Target sync read throughput (e.g. 500m, 2g). Auto-tunes the first read pass.")
	parser.add_argument("-V","--version", action="version", version=f"%(prog)s {version} @ {COMMIT_DATE} with teeLogger {Tee_Logger.version} {' and numpy ' if numpy_available else ''} by pan@zopyr.us")
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
	tl = Tee_Logger.teeLogger(systemLogFileDir=args.log_directory,programName='iotest',compressLogAfterMonths=1,deleteLogAfterYears=3,suppressPrintout=args.stealth,noLog=args.no_log,in_place_compression=True)
	tl.info(f'Arguments: {vars(args)}')
	file_size_specified = args.file_size is not None
	file_count_specified = args.file_count is not None
	process_count_specified = args.process_count is not None

	args.file_size = parse_size_arg(args.file_size if file_size_specified else default_file_size_str, unitless_as_mb=True)
	args.file_count = args.file_count if file_count_specified else default_file_count
	args.process_count = args.process_count if process_count_specified else default_process_count

	if args.total_size is not None:
		total_size = parse_size_arg(args.total_size, unitless_as_mb=True)
		if total_size <= 0:
			raise ValueError("total_size must be greater than 0")
		if file_size_specified and file_count_specified and process_count_specified:
			tl.teelog("Warning | iotest: -ts/--total_size is ignored because -fs, -fc, and -pc/-t are all explicitly set.", level='warning')
		else:
			if not file_count_specified:
				denominator = args.file_size * args.process_count
				if denominator <= 0:
					raise ValueError("Cannot calculate file_count because file_size * process_count is invalid")
				args.file_count = max(total_size // denominator, 1)
				tl.teelog(f"Adjusted -fc to {args.file_count} to fit total_size {format_bytes(total_size)}B", level='info')
			elif not file_size_specified:
				denominator = args.file_count * args.process_count
				if denominator <= 0:
					raise ValueError("Cannot calculate file_size because file_count * process_count is invalid")
				args.file_size = max(total_size // denominator, 1)
				tl.teelog(f"Adjusted -fs to {format_bytes(args.file_size)}B to fit total_size {format_bytes(total_size)}B", level='info')
			elif not process_count_specified:
				denominator = args.file_size * args.file_count
				if denominator <= 0:
					raise ValueError("Cannot calculate process_count because file_size * file_count is invalid")
				args.process_count = max(total_size // denominator, 1)
				tl.teelog(f"Adjusted -t/-pc to {args.process_count} to fit total_size {format_bytes(total_size)}B", level='info')

	modes = parse_modes(args.mode)
	write_target = None
	read_target = None
	if args.write_target_throughput is not None:
		if not supports_write_throughput_tuning(modes):
			raise ValueError("-wtpt/--write_target_throughput requires a mode with a write pass (write, rw, comprehensive, simultaneous, etc.)")
		write_target = parse_size_arg(args.write_target_throughput, unitless_as_mb=True)
		if write_target <= 0:
			raise ValueError("write_target_throughput must be greater than 0")
	if args.read_target_throughput is not None:
		if not supports_read_throughput_tuning(modes):
			raise ValueError("-rtpt/--read_target_throughput requires a mode with a read pass (read, rw, comprehensive, simultaneous, etc.)")
		read_target = parse_size_arg(args.read_target_throughput, unitless_as_mb=True)
		if read_target <= 0:
			raise ValueError("read_target_throughput must be greater than 0")
	throughput_tuning = write_target is not None or read_target is not None
	# if no directory is provided, use the current directory
	if args.directory is None:
		args.directory = [os.getcwd()]
	if args.stealth:
		tl.info(f'Running in {modes} modes...')
		tl.info(f'File size: {format_bytes(args.file_size)}B')
		tl.info(f'Number of files: {args.file_count}')
		tl.info(f'Number of processes: {args.process_count}')
		if write_target is not None:
			tl.info(f'Write target throughput: {format_bytes(write_target)}B/s')
		if read_target is not None:
			tl.info(f'Read target throughput: {format_bytes(read_target)}B/s')
		if 'benchmark' not in modes and len(modes) == 1:
			tl.info(f'Writing to {", ".join(args.directory)}')
	else:
		tl.teeprint(f'Running in {modes} modes...')
		tl.teeprint(f'File size: {format_bytes(args.file_size)}B')
		tl.teeprint(f'Number of files: {args.file_count}')
		tl.teeprint(f'Number of processes: {args.process_count}')
		if write_target is not None:
			tl.teeprint(f'Write target throughput: {format_bytes(write_target)}B/s')
		if read_target is not None:
			tl.teeprint(f'Read target throughput: {format_bytes(read_target)}B/s')
		if 'benchmark' not in modes and len(modes) == 1:
			tl.teeprint(f'Writing to {", ".join(args.directory)}')
	if throughput_tuning:
		tune_for_targets(args.file_size, args.file_count, args.process_count, args.directory, modes,
			args.quiet, args.zeros, tl, write_target, read_target, stealth=args.stealth,
			message_end_point_address=args.message_end_point_address, no_report=args.no_report,
			threshold_to_report_anomaly=args.threshold_to_report_anomaly, benchmarkTime=args.benchmark_time)
	else:
		main(args.file_size, args.file_count, args.process_count, args.directory,modes,args.quiet,args.zeros,tl=tl,stealth=args.stealth,message_end_point_address=args.message_end_point_address,no_report=args.no_report,threshold_to_report_anomaly=args.threshold_to_report_anomaly,benchmarkTime=args.benchmark_time)
	
if __name__ == "__main__":
	climain()