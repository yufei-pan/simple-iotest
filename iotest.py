#!/usr/bin/env python3
import argparse
import array
import csv
import datetime
import functools
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import re
import shutil
import socket
import stat
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib

try:
	import fcntl
except ImportError:
	# Windows has no fcntl
	fcntl = None
from multiprocessing import Manager, Process

numpy_available = False
try:
	import numpy as np # type: ignore
	from numpy.random import SFC64, Generator # type: ignore
	numpy_available = True
except ImportError:
	numpy_available = False

version = '3.70.0'
COMMIT_DATE = '2026-10-18'
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

def version_at_least(version_text, minimum):
	# compare as integer tuples, not floats: 6.100 is newer than 6.34; a version that does not parse raises ValueError
	return tuple(int(part) for part in version_text.split('.')) >= tuple(int(part) for part in minimum.split('.'))

# ------------------------------
try:
	import Tee_Logger # type: ignore
	if not version_at_least(Tee_Logger.version, '6.34'):
		raise ImportError(f"Tee_Logger {Tee_Logger.version} is older than 6.34")
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
					print('Using inline logger, forcing noLog...', file=sys.stderr)
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

GEN_CHUNK = 16 * 2**20

def almost_urandom(n):
	# fill one buffer chunk by chunk: generating it in one go briefly holds a
	# second full-size copy (numpy array + .tobytes(), or big int + to_bytes)
	buf = bytearray(n)
	view = memoryview(buf)
	rng_sf = Generator(SFC64()) if numpy_available else None
	for offset in range(0, n, GEN_CHUNK):
		size = min(GEN_CHUNK, n - offset)
		if rng_sf is not None:
			view[offset:offset + size] = rng_sf.bit_generator.random_raw((size + 7) // 8).view('uint8')[:size]
		else:
			view[offset:offset + size] = random.getrandbits(8 * size).to_bytes(size, 'big')
	return buf

UNIQUE_STAMP_INTERVAL = 512
UNIQUE_TAG = struct.Struct('<QQQ')

def stamp_unique(buf, worker_id, file_index, first_block=0):
	# Workers reuse one buffer for all their files. Overwrite the start of every
	# 512-byte block with (file, block, worker) so no two blocks of the run match,
	# which dedup storage could otherwise keep a single physical copy of. The block
	# number matters with -z, where everything after the tag is zeros. The file
	# index goes first so a final block shorter than the tag still differs.
	# buf starts at block first_block of the file (with -bs it holds one chunk).
	size = len(buf)
	full_blocks = size // UNIQUE_STAMP_INTERVAL
	if numpy_available and full_blocks:
		tags = np.empty((full_blocks, 3), dtype='<u8')
		tags[:, 0] = file_index
		tags[:, 1] = np.arange(first_block, first_block + full_blocks, dtype='<u8')
		tags[:, 2] = worker_id
		blocks = np.frombuffer(buf, dtype=np.uint8)[:full_blocks * UNIQUE_STAMP_INTERVAL]
		blocks.reshape(full_blocks, UNIQUE_STAMP_INTERVAL)[:, :UNIQUE_TAG.size] = tags.view(np.uint8).reshape(full_blocks, UNIQUE_TAG.size)
	else:
		for block in range(full_blocks):
			UNIQUE_TAG.pack_into(buf, block * UNIQUE_STAMP_INTERVAL, file_index, first_block + block, worker_id)
	tail = size - full_blocks * UNIQUE_STAMP_INTERVAL
	if tail:
		start = full_blocks * UNIQUE_STAMP_INTERVAL
		length = min(tail, UNIQUE_TAG.size)
		memoryview(buf)[start:start + length] = UNIQUE_TAG.pack(file_index, first_block + full_blocks, worker_id)[:length]

def stamp_chunk(chunk, offset, worker_id, file_index):
	# with -bs a file is stamped one chunk at a time, at its offset in the file; the tags land
	# at the offsets whole-file stamping uses, so every 512-byte block has its own tag, but the
	# reused block buffer keeps the earlier chunks' tag bytes (and the previous file's CRC
	# trailer) elsewhere, so with -z -u -bs the bytes between tags are not all zeros. A chunk
	# that starts inside a 512-byte block (a -bs that is not a multiple of 512) first finishes
	# the tag the previous one began
	block, into = divmod(offset, UNIQUE_STAMP_INTERVAL)
	if into:
		rest = UNIQUE_TAG.pack(file_index, block, worker_id)[into:into + len(chunk)]
		chunk[:len(rest)] = rest
		skip = UNIQUE_STAMP_INTERVAL - into
		if skip >= len(chunk):
			return
		chunk = chunk[skip:]
		block += 1
	stamp_unique(chunk, worker_id, file_index, first_block=block)

def new_stamp_id():
	# random rather than the PID: PIDs get reused, and with -z a reused one would
	# rebuild the exact blocks of files an earlier run left on the same storage
	return int.from_bytes(os.urandom(8), 'little')

def effective_block_size(block_size, file_size):
	# a block at least as large as the file is one call per file, the same as no block size
	if block_size is None or block_size >= file_size:
		return None
	return int(block_size)

VERIFY_TRAILER = struct.Struct('<I')

class CrcTrailer:
	# --verify: a file's last 4 bytes hold the CRC-32 of every byte before them, so any later
	# read can check the file without stored state. Fed one file's chunks in order: with -bs
	# the CRC runs across chunks, and a block smaller than the trailer splits it.
	def __init__(self, file_size):
		self.data_end = file_size - VERIFY_TRAILER.size
		self.crc = 0
		self.stored = bytearray(VERIFY_TRAILER.size)

	def add(self, chunk, offset):
		# counts the chunk's data bytes; returns where its trailer part starts in it and in the trailer
		split = min(max(self.data_end - offset, 0), len(chunk))
		self.crc = zlib.crc32(chunk[:split], self.crc) & 0xffffffff
		return split, offset + split - self.data_end

	def seal(self, chunk, offset):
		split, into = self.add(chunk, offset)
		if split < len(chunk):
			# every byte before the trailer has been counted by now
			chunk[split:] = VERIFY_TRAILER.pack(self.crc)[into:into + len(chunk) - split]

	def check(self, chunk, offset):
		split, into = self.add(chunk, offset)
		if split < len(chunk):
			self.stored[into:into + len(chunk) - split] = chunk[split:]

	def mismatch(self, file_name):
		stored = VERIFY_TRAILER.unpack(self.stored)[0]
		if stored == self.crc:
			return None
		return f"verify failed: {file_name}: CRC-32 mismatch (stored 0x{stored:08x}, computed 0x{self.crc:08x})"

class FileContent:
	# what a worker writes, reused for all its files: the whole file, or one block with -bs.
	# Made on first use (or by prepare), so read workers whose files all exist never hold it.
	def __init__(self, file_size, zeros, unique, stamp_id, block_size=None, verify=False):
		self.file_size = file_size
		self.zeros = zeros
		self.unique = unique
		self.stamp_id = stamp_id
		self.block_size = block_size
		self.verify = verify
		self.content = None

	def prepare(self):
		if self.content is None:
			self.content = make_file_content(self.block_size or self.file_size, self.zeros, self.unique)

	def write(self, file_name, index, quiet=False):
		self.prepare()
		stamp = None
		seal = None
		if self.unique and self.block_size:
			# the block is reused for the whole file, so each chunk is stamped as it goes out, inside the timed write
			stamp = functools.partial(stamp_chunk, worker_id=self.stamp_id, file_index=index)
		elif self.unique:
			stamp_unique(self.content, self.stamp_id, index)
		# the CRC covers the stamps, so it comes after them: per chunk inside the timed write, or here before it
		if self.verify and self.block_size:
			seal = CrcTrailer(self.file_size).seal
		elif self.verify:
			CrcTrailer(self.file_size).seal(memoryview(self.content), 0)
		return create_file(file_name, self.content, self.file_size, quiet=quiet, block_size=self.block_size, stamp=stamp, seal=seal)

def make_file_content(file_size, zeros, unique):
	if zeros:
		# stamp_unique edits the buffer in place, so -u needs a mutable one
		return bytearray(file_size) if unique else b'\x00' * file_size
	return almost_urandom(file_size)

# A single read()/write() may transfer less than asked: Linux caps one call at
# this many bytes (MAX_RW_COUNT, 2 GiB - 4 KiB) and macOS rejects anything over
# INT_MAX with EINVAL, so large files are moved in chunks of at most this size.
MAX_IO_CHUNK = 0x7ffff000

def disable_os_cache(fd):
	# macOS has no posix_fadvise; F_NOCACHE keeps its pages out of the cache instead
	# (set on the write fd too, so a read cannot hit pages the write left cached)
	no_cache = getattr(fcntl, 'F_NOCACHE', None)
	if no_cache is None:
		return
	try:
		fcntl.fcntl(fd, no_cache, 1)
	except OSError:
		pass

def file_chunks(buffer, file_size, block_size):
	# (offset in the file, view) for each part of the file one write/read is asked for: the
	# whole file from a file-size buffer, or with -bs block_size bytes at a time from one block
	view = memoryview(buffer)
	# max(): an empty file still needs a non-zero range() step
	step = block_size or max(file_size, 1)
	for offset in range(0, file_size, step):
		yield offset, view[:min(step, file_size - offset)]

def write_chunk(f, chunk, offset, file_name, file_size, quiet):
	written = 0
	while written < len(chunk):
		part = chunk[written:written + MAX_IO_CHUNK]
		try:
			if os.name == 'posix':
				n = os.writev(f.fileno(), [part])
			else:
				n = os.write(f.fileno(), part)
		except OSError:
			worker_log('Failed to write using os.writev, trying f.write', level='warning', quiet=quiet)
			n = f.write(part)
		if not n:
			raise OSError(f"{file_name}: write stopped after {offset + written} of {file_size} bytes")
		written += n

def read_chunk(f, chunk, offset, file_name, file_size, quiet):
	got = 0
	while got < len(chunk):
		part = chunk[got:got + MAX_IO_CHUNK]
		try:
			if os.name == 'posix':
				n = os.readv(f.fileno(),[part])
			else:
				n = f.readinto(part)
		except OSError:
			worker_log('Failed to read using os.readv, trying f.readinto', level='warning', quiet=quiet)
			n = f.readinto(part)
		if not n:
			raise OSError(f"{file_name}: read hit end of file after {offset + got} of {file_size} bytes")
		got += n

def create_file(file_name, file_content, file_size, quiet=False, block_size=None, stamp=None, seal=None):
	# with block_size, file_content is one block written again and again; stamp(chunk, offset)
	# marks each chunk for -u and then seal(chunk, offset) adds it to the --verify CRC, just before it is written
	try:
		with open(file_name, "wb", buffering=0) as f:
			disable_os_cache(f.fileno())
			start_write_time = time.perf_counter()
			for offset, chunk in file_chunks(file_content, file_size, block_size):
				if stamp is not None:
					stamp(chunk, offset)
				if seal is not None:
					seal(chunk, offset)
				write_chunk(f, chunk, offset, file_name, file_size, quiet)
			os.fsync(f.fileno())
			end_write_time = time.perf_counter()
			# after the sync the pages are clean and can be dropped; before the write the file is empty
			try:
				if os.name == 'posix':
					os.posix_fadvise(f.fileno(), 0, file_size, os.POSIX_FADV_DONTNEED)
			except (AttributeError, OSError):
				# AttributeError: macOS is posix but has no posix_fadvise
				worker_log('Failed to posix_fadvise', level='warning', quiet=quiet)
			return start_write_time,end_write_time
	except Exception as e:
		import traceback
		worker_log(str(e), level='error', quiet=quiet)
		worker_log(traceback.format_exc(), level='error', quiet=quiet)
		return None,f'{type(e).__name__}: {e}'

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
		return None,f'{type(e).__name__}: {e}'

def file_needs_recreate(file_name, file_size):
	return not os.path.isfile(file_name) or os.path.getsize(file_name) != file_size

def read_file(file_name, file_content, file_size, quiet=False, buffer=None, recreate=None, block_size=None, verify=False):
	# workers pass one reused buffer; a fresh one costs a zero-fill per read (~150 ms/GiB)
	b = buffer if buffer is not None else bytearray(block_size or file_size)
	# check if file exists and size is correct
	try:
		if file_needs_recreate(file_name, file_size):
			# file does not exist or size is wrong, create it
			worker_log(f"File {file_name} does not exist or size is wrong, creating it...", level='error', quiet=quiet)
			# workers pass recreate(file_name), which writes the file as they write theirs: read-only
			# workers hold no content until a file actually has to be recreated
			if recreate is not None:
				created = recreate(file_name)
			else:
				created = create_file(file_name, file_content, file_size, quiet=quiet, block_size=block_size)
			if created[0] is None:
				# report why the file is missing, not the read error that would follow
				return None, f"recreating missing {file_name}: {created[1]}"

		trailer = CrcTrailer(file_size) if verify else None
		with open(file_name, "rb", buffering=0) as f:
			disable_os_cache(f.fileno())
			try:
				if os.name == 'posix':
					os.posix_fadvise(f.fileno(), 0, file_size, os.POSIX_FADV_DONTNEED)
			except (AttributeError, OSError):
				# AttributeError: macOS is posix but has no posix_fadvise
				worker_log('Failed to posix_fadvise', level='warning', quiet=quiet)
			start_read_time = time.perf_counter()
			for offset, chunk in file_chunks(b, file_size, block_size):
				read_chunk(f, chunk, offset, file_name, file_size, quiet)
				if trailer is not None and block_size:
					# the next chunk reuses the buffer, so the CRC is taken now, inside the timed read
					trailer.check(chunk, offset)
			end_read_time = time.perf_counter()
		if trailer is not None:
			if not block_size:
				# the whole file is in the buffer: check it after the timer, as the write sealed it before
				trailer.check(memoryview(b)[:file_size], 0)
			mismatch = trailer.mismatch(file_name)
			if mismatch:
				worker_log(mismatch, level='error', quiet=quiet)
				return None, mismatch
		return start_read_time,end_read_time
	except Exception as e:
		import traceback
		worker_log(str(e), level='error', quiet=quiet)
		worker_log(traceback.format_exc(), level='error', quiet=quiet)
		return None,f'{type(e).__name__}: {e}'
	

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
		return None,f'{type(e).__name__}: {e}'
	
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
		return None,f'{type(e).__name__}: {e}',0

META_PHASES = ('create', 'stat', 'unlink')

def meta_dir_path(directory, counter_name):
	return os.path.join(directory, str(counter_name), 'meta')

def meta_file_path(directory, counter_name, index, file_count):
	return os.path.join(meta_dir_path(directory, counter_name), f"meta_{str(index + 1).zfill(len(str(file_count)))}")

def meta_file(phase, file_name, quiet=False):
	start = time.perf_counter()
	try:
		if phase == 'create':
			os.close(os.open(file_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
		elif phase == 'stat':
			os.stat(file_name)
		else:
			os.unlink(file_name)
		return start, time.perf_counter()
	except Exception as e:
		worker_log(f'{phase} {file_name}: {e}', level='error', quiet=quiet)
		return None, f'{type(e).__name__}: {e}'

def remove_meta_leftovers(meta_dir):
	# untimed: files an interrupted run left behind would make O_EXCL creates fail
	try:
		names = os.listdir(meta_dir)
	except OSError:
		return
	for name in names:
		if name.startswith('meta_'):
			try:
				os.unlink(os.path.join(meta_dir, name))
			except OSError:
				pass

def remove_meta_dir(meta_dir):
	# the counter dir above it was made for the meta dir; rmdir leaves it alone if it still holds files
	try:
		os.rmdir(meta_dir)
		os.rmdir(os.path.dirname(meta_dir))
	except OSError:
		pass

def op_time(op_result):
	# helpers return (start, end, ...) on success and (None, error_text, ...) on failure;
	# a failed op keeps its error text in place of a duration so it never enters the timing stats
	start, end = op_result[0], op_result[1]
	if start is None:
		return end
	return end - start

def format_op(op, spec=''):
	if isinstance(op, str):
		return 'FAILED'
	return f'{op:{spec}} s'

def int_to_color(n, brightness_threshold=500):
	hash_value = hash(str(n))
	r = (hash_value >> 16) & 0xFF
	g = (hash_value >> 8) & 0xFF
	b = hash_value & 0xFF
	if (r + g + b) < brightness_threshold:
		return int_to_color(hash_value, brightness_threshold)
	return (r, g, b)

# op names; workers store an op as its index here, in one byte (the timeline's op column, random's result kinds)
OP_NAMES = ('write', 'read', 'move', 'stat', 'index', 'create', 'unlink')
OP_CODES = {op: code for code, op in enumerate(OP_NAMES)}

def results_record(samples, kinds=None):
	# one worker's results of one pass, handed to the parent as one compact record: the durations as the
	# bytes of an array('d'), a failed op's as NaN with its error text in errors, and each op's kind (an
	# OP_CODES byte) when the pass mixes them. Built after the last op, from the list the ops appended to
	try:
		durations = array.array('d', samples)
		errors = {}
	except TypeError:
		# some op failed: op_time() gave its error text
		errors = {position: sample for position, sample in enumerate(samples) if isinstance(sample, str)}
		durations = array.array('d', (math.nan if isinstance(sample, str) else sample for sample in samples))
	return (durations.tobytes(), None if kinds is None else bytes(kinds), errors)

def result_samples(records, kind=None, every=1, offset=0):
	# the records' ops in arrival order as op_time() gave them (the duration, or the error text of a failed
	# op): all of them, every n-th from offset (comprehensive's write, move, stat and read of each file),
	# or those of one kind (random). Only the selected ops are converted, so taking a pass key by key
	# converts each duration once
	for durations, kinds, errors in records:
		durations = memoryview(durations).cast('d')
		if kind is None:
			samples = durations[offset::every].tolist()
			for position, error in errors.items():
				if position >= offset and (position - offset) % every == 0:
					samples[(position - offset) // every] = error
		else:
			samples = [errors[position] if position in errors else durations[position]
				for position, op_kind in enumerate(kinds) if op_kind == kind]
		yield from samples

class OpTimeline:
	# one worker's ops of one pass in typed arrays (21 bytes an op), handed to the parent as one record
	def __init__(self, counter, base, file_count):
		self.counter = counter
		self.base = base
		self.ops = array.array('B')
		self.file_indexes = array.array('I' if file_count <= 0xFFFFFFFF else 'Q')
		self.starts = array.array('d')
		self.durations = array.array('d')
		# position -> error text of the failed ops, whose duration is NaN
		self.errors = {}

	def add(self, op, file_index, attempt_start, result):
		# seconds from the pass's start barrier; a failed op has no start of its own, so it gets when the attempt began
		start, end = result[0], result[1]
		if start is None:
			self.errors[len(self.starts)] = end
			self.starts.append(attempt_start - self.base)
			self.durations.append(math.nan)
		else:
			self.starts.append(start - self.base)
			self.durations.append(end - start)
		self.ops.append(OP_CODES[op])
		self.file_indexes.append(file_index)

	def record(self):
		# plain bytes, which the manager can hold whatever the start method. Pickle copies an array's data
		# to bytes and keeps that copy until the whole transfer ends, so as bytes a pass's records take one
		# copy less in the manager and in the parent
		return (self.counter, self.file_indexes.typecode, self.ops.tobytes(), self.file_indexes.tobytes(),
			self.starts.tobytes(), self.durations.tobytes(), self.errors)

def worker_color_log(color, msg, quiet=False):
	# each worker prints in its own color, so interleaved lines stay apart
	r, g, b = color
	worker_log(f'\033[38;2;{r};{g};{b}m{msg}\033[0m', quiet=quiet)

def wait_for_barrier(start_time):
	# one long sleep until about 1 s before the barrier, then the 1 ms loop for the precise start
	ahead = start_time - time.perf_counter() - 1
	if ahead > 0:
		time.sleep(ahead)
	while time.perf_counter() < start_time:
		time.sleep(0.001)

def worker(file_count, file_size, directory, results, mode, counter, quiet, zeros, thread_start_time, unique=False, recreates=False, late_starts=None, progress_to_stderr=False, timeline=None, block_size=None, verify=False):
	# the ops append to a list as they go; it becomes one compact record after the last
	local_results = []
	local_kinds = [] if mode == 'random' else None
	# only with --timeline; without it the loop below does no timeline work beyond a None check
	local_timeline = OpTimeline(counter, thread_start_time, file_count) if timeline is not None else None
	log = functools.partial(worker_color_log, int_to_color(os.getpid()), quiet=quiet)
	log(f'Worker {counter} scheduled to start at {thread_start_time-time.perf_counter():.4f} later in {mode} mode.')
	# read workers create content only if a file must be recreated; index never needs any
	content = FileContent(file_size, zeros, unique, new_stamp_id(), block_size, verify)
	if mode in ('write', 'random', 'comprehensive') or (recreates and mode == 'read'):
		# make it now, before the start barrier, not in the timed window
		content.prepare()
	read_buffer = bytearray(block_size or file_size) if mode in ('read', 'random', 'comprehensive') else None
	is_meta = mode.startswith('meta_')
	if is_meta:
		meta_dir = meta_dir_path(directory, counter)
		if mode == 'meta_create':
			os.makedirs(meta_dir, exist_ok=True)
			remove_meta_leftovers(meta_dir)
	if time.perf_counter() > thread_start_time:
		worker_log(f'Worker {counter} started late, expected start time {thread_start_time}, actual start time {time.perf_counter()}', level='error', quiet=quiet)
		if late_starts is not None:
			late_starts.append(time.perf_counter() - thread_start_time)
	log(f'Worker {counter} primed, waiting for {thread_start_time-time.perf_counter():.4f} seconds.')
	# wait until thread_start_time
	wait_for_barrier(thread_start_time)
	if not is_meta:
		os.makedirs(os.path.join(directory,str(counter)), exist_ok=True)
	reportInterval = max(file_count // 10,1)
	for i in range(file_count):
		# print a digit every 10% of the files
		if i % reportInterval == 0 and quiet:
				# stderr when stdout carries the JSON summary
				print(f'\r{i//reportInterval}',end='',flush=True,file=sys.stderr if progress_to_stderr else None)
		file_name = meta_file_path(directory, counter, i, file_count) if is_meta else worker_file_path(directory, counter, i, file_count)
		write_file = functools.partial(content.write, index=i, quiet=quiet)
		file_index = i + 1
		i = str(file_index).zfill(len(str(file_count)))
		attempt = time.perf_counter() if local_timeline is not None else None
		if mode == 'write':
			op_result = write_file(file_name)
			if local_timeline is not None:
				local_timeline.add('write', file_index, attempt, op_result)
			write_time = op_time(op_result)
			local_results.append(write_time)
			log(f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tWrote\t{format_op(write_time)}")
		elif mode == 'read':
			op_result = read_file(file_name, None, file_size, quiet, buffer=read_buffer, recreate=write_file, block_size=block_size, verify=verify)
			if local_timeline is not None:
				local_timeline.add('read', file_index, attempt, op_result)
			read_time = op_time(op_result)
			local_results.append(read_time)
			log(f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tRead\t{format_op(read_time)}")
		elif mode == 'random':
			# here we start a random read or write
			if bool(random.getrandbits(1)):
				op_result = write_file(file_name)
				if local_timeline is not None:
					local_timeline.add('write', file_index, attempt, op_result)
				write_time = op_time(op_result)
				local_results.append(write_time)
				local_kinds.append(OP_CODES['write'])
				log(f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tWrote\t{format_op(write_time)}")
			else:
				op_result = read_file(file_name, None, file_size, quiet, buffer=read_buffer, recreate=write_file, block_size=block_size, verify=verify)
				if local_timeline is not None:
					local_timeline.add('read', file_index, attempt, op_result)
				read_time = op_time(op_result)
				local_results.append(read_time)
				local_kinds.append(OP_CODES['read'])
				log(f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tRead\t{format_op(read_time)}")
		elif mode == 'comprehensive':
			resultList = []
			moved_name = file_name + ".moved"
			op_result = write_file(file_name)
			if local_timeline is not None:
				local_timeline.add('write', file_index, attempt, op_result)
			resultList.append(op_time(op_result))
			attempt = time.perf_counter() if local_timeline is not None else None
			op_result = move_file(file_name, moved_name, quiet=quiet)
			if local_timeline is not None:
				local_timeline.add('move', file_index, attempt, op_result)
			resultList.append(op_time(op_result))
			attempt = time.perf_counter() if local_timeline is not None else None
			stat_result = stat_file(moved_name, quiet=quiet)
			if local_timeline is not None:
				local_timeline.add('stat', file_index, attempt, stat_result)
			size = stat_result[2]
			if size != file_size:
				worker_log(f'{bcolors.critical}File {file_name} size is wrong, expected {file_size}, got {size}\033[0m', level='error', quiet=quiet)
			resultList.append(op_time(stat_result))
			attempt = time.perf_counter() if local_timeline is not None else None
			op_result = read_file(moved_name, None, file_size, quiet, buffer=read_buffer, recreate=write_file, block_size=block_size, verify=verify)
			if local_timeline is not None:
				local_timeline.add('read', file_index, attempt, op_result)
			resultList.append(op_time(op_result))
			try:
				os.remove(moved_name)
			except OSError:
				pass
			log(f'[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tW: {format_op(resultList[0], ".4f")} M: {format_op(resultList[1], ".4f")} S: {format_op(resultList[2], ".4f")} R: {format_op(resultList[3], ".4f")}')
			local_results.extend(resultList)
		elif mode == 'index':
			# This will test the indexing performance of the filesystem
			op_result = index_file(file_name, quiet)
			if local_timeline is not None:
				local_timeline.add('index', file_index, attempt, op_result)
			index_time = op_time(op_result)
			local_results.append(index_time)
			log(f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\tIndexed\t{format_op(index_time)}")
		elif is_meta:
			phase = mode[len('meta_'):]
			op_result = meta_file(phase, file_name, quiet)
			if local_timeline is not None:
				local_timeline.add(phase, file_index, attempt, op_result)
			meta_time = op_time(op_result)
			local_results.append(meta_time)
			log(f"[Process {os.getpid()}]\tFile {i}/{file_count}:\t{file_name}\t{phase.capitalize()}\t{format_op(meta_time)}")
	if mode == 'meta_unlink':
		remove_meta_dir(meta_dir)
	log(f'Worker {counter} finished.')
	results.append(results_record(local_results, local_kinds))
	if local_timeline is not None:
		# one record per worker and pass, so the manager sees one append, not one per op
		timeline.append(local_timeline.record())

# worker directories are named by the counter at this width whatever the process count, so a run at another
# count uses the same workers' files: a smaller count is the first workers of a larger one
WORKER_NAME_DIGITS = 4

def worker_counter_name(counter):
	return str(counter).zfill(WORKER_NAME_DIGITS)

def worker_file_path(directory, counter_name, index, file_count):
	return os.path.join(directory, str(counter_name), f"temp_{str(index + 1).zfill(len(str(file_count)))}.bin")

def worker_dirs(directories, process_count):
	# (directory, counter_name) of every plain worker
	for counter in range(process_count):
		yield directories[counter % len(directories)], worker_counter_name(counter)

def worker_files(directories, process_count, file_count):
	# (directory, counter_name, path) of every temp file the plain workers use
	for directory, counter_name in worker_dirs(directories, process_count):
		for index in range(file_count):
			yield directory, counter_name, worker_file_path(directory, counter_name, index, file_count)

def existing_size(path):
	try:
		st = os.stat(path)
	except OSError:
		return 0
	return st.st_size if stat.S_ISREG(st.st_mode) else 0

def scan_worker_files(directories, process_count, file_count, file_size):
	# the plain workers' temp files as the run finds them, one os.stat per file: per worker, in counter order,
	# (bytes its files hold, how many of them are missing or not file_size bytes). Names do not depend on the
	# process count, so the first n entries describe a run at n workers
	scan = []
	for directory, counter_name in worker_dirs(directories, process_count):
		existing = wrong = 0
		for index in range(file_count):
			size = existing_size(worker_file_path(directory, counter_name, index, file_count))
			existing += size
			if size != file_size:
				wrong += 1
		scan.append((existing, wrong))
	return scan

FREE_SPACE_MARGIN = 0.95

COPY_ON_WRITE_FILESYSTEMS = ('btrfs', 'zfs')

def space_needed(modes, file_size, file_count, process_count, directories,
		simultaneous_write_count=None, layout_worker_count=None, skip_layout=False, mountinfo_lines=None, scan=None):
	# peak bytes each directory receives during the run; scan is scan_worker_files() of at least process_count
	# workers, made here when not given
	if mountinfo_lines is None:
		mountinfo_lines = read_mountinfo()
	# needed assumes a rewrite frees the old file first; cow_extra is what copy-on-write
	# filesystems need on top when a snapshot keeps the old blocks allocated
	needed = {d: 0 for d in directories}
	cow_extra = {d: 0 for d in directories}
	rewrites = 'write' in modes or 'random' in modes
	if rewrites or 'read' in modes:
		# these files stay; read only recreates a missing or wrong-size file, at full size
		cow = {d: (find_mount(os.path.realpath(d), mountinfo_lines) or ('',))[0] in COPY_ON_WRITE_FILESYSTEMS
			for d in directories}
		if scan is None:
			scan = scan_worker_files(directories, process_count, file_count, file_size)
		for counter in range(process_count):
			directory = directories[counter % len(directories)]
			existing, wrong = scan[counter]
			if rewrites:
				needed[directory] += file_size * file_count - existing
				if cow[directory]:
					cow_extra[directory] += existing
			else:
				needed[directory] += file_size * wrong
	# comprehensive deletes each file after reading it and simultaneous removes its
	# tree when done, so only the larger of the two is ever on disk at once
	transient = {d: 0 for d in directories}
	if 'comprehensive' in modes:
		for counter in range(process_count):
			transient[directories[counter % len(directories)]] += file_size
	# skip_layout means the tuner already laid out the read files after checking space for
	# them plus its largest writer count; attempts overwrite the same writer files
	if 'simultaneous' in modes and not skip_layout:
		simul = {d: 0 for d in directories}
		write_count = simultaneous_write_count if simultaneous_write_count is not None else process_count // 2
		layout_count = layout_worker_count if layout_worker_count is not None else process_count
		for count in (write_count, layout_count):
			for counter in range(count):
				simul[directories[counter % len(directories)]] += file_size * file_count
		for d in directories:
			transient[d] = max(transient[d], simul[d])
	return {d: needed[d] + transient[d] for d in directories}, cow_extra

def space_pool(directory, mountinfo_lines):
	# directories drawing on the same free space; btrfs subvolumes and ZFS datasets
	# each get their own st_dev but share their filesystem's or pool's free space
	mount = find_mount(os.path.realpath(directory), mountinfo_lines)
	if mount:
		fstype, source = mount
		if fstype == 'btrfs':
			return ('btrfs', source)
		if fstype == 'zfs':
			return ('zfs', source.split('/')[0])
	return ('dev', os.stat(directory).st_dev)

def check_free_space(needed_by_dir, tl, mountinfo_lines=None, cow_extra=None):
	if mountinfo_lines is None:
		mountinfo_lines = read_mountinfo()
	spare = f"keeping {100 - FREE_SPACE_MARGIN * 100:.0f}% spare"
	enough = True
	pools = {}
	for directory, needed in needed_by_dir.items():
		free = shutil.disk_usage(directory).free
		# each directory on its own: a ZFS quota or btrfs qgroup can leave it less room than its pool
		if needed > free * FREE_SPACE_MARGIN:
			tl.teeerror(f"Not enough free space for {directory}: this run needs {format_bytes(needed)}B, "
				f"{format_bytes(free)}B free ({spare}).")
			enough = False
		elif cow_extra and needed + cow_extra.get(directory, 0) > free * FREE_SPACE_MARGIN:
			tl.teelog(f"Warning: {directory} is copy-on-write; rewriting the existing test files frees their "
				f"old blocks only if no snapshot holds them. If one does, this run needs up to "
				f"{format_bytes(needed + cow_extra[directory])}B ({format_bytes(free)}B free).", level='warning')
		pool = pools.setdefault(space_pool(directory, mountinfo_lines), [0, 0, [], 0])
		pool[0] += needed
		pool[1] = max(pool[1], free)
		pool[2].append(directory)
		pool[3] += cow_extra.get(directory, 0) if cow_extra else 0
	# ...and together, when several directories draw on one pool
	for needed, free, dirs, extra in pools.values():
		if len(dirs) < 2:
			continue
		if needed > free * FREE_SPACE_MARGIN:
			tl.teeerror(f"Not enough free space for {', '.join(dirs)}, which share their free space: "
				f"this run needs {format_bytes(needed)}B, {format_bytes(free)}B free ({spare}).")
			enough = False
		elif extra and needed + extra > free * FREE_SPACE_MARGIN:
			tl.teelog(f"Warning: {', '.join(dirs)} share copy-on-write free space; rewriting the existing "
				f"test files frees their old blocks only if no snapshot holds them. If one does, this run "
				f"needs up to {format_bytes(needed + extra)}B ({format_bytes(free)}B free).", level='warning')
	if not enough:
		tl.teeerror("Nothing was written. Reduce -fs, -fc or -pc (or use -ts) and try again.")
	return enough

# the largest process count run_iotest() ran plain workers at in this process. Every run's workers are the
# first that many, so -cl after tuning and the leftover hint cover the files this run used, and not those
# of other runs' workers beyond them
largest_worker_run = 0

def note_worker_run(process_count):
	global largest_worker_run
	largest_worker_run = max(largest_worker_run, process_count)

def remove_worker_files(entries):
	# remove only these files, then their worker dirs if now empty
	counter_dirs = set()
	for directory, counter_name, path in entries:
		counter_dirs.add(os.path.join(directory, counter_name))
		try:
			os.remove(path)
		except OSError:
			pass
	for counter_dir in counter_dirs:
		try:
			os.rmdir(counter_dir)
		except OSError:
			pass

def cleanup_worker_files(directories, process_count, file_count):
	remove_worker_files(worker_files(directories, process_count, file_count))

def cleanup_meta_dirs(directories, process_count):
	for directory, counter_name in worker_dirs(directories, process_count):
		meta_dir = meta_dir_path(directory, counter_name)
		remove_meta_leftovers(meta_dir)
		remove_meta_dir(meta_dir)
		try:
			os.rmdir(os.path.join(directory, counter_name))
		except OSError:
			pass

def leftover_hint(directories, worker_count, file_count):
	# the files of the first worker_count workers, going through them instead of holding every path
	count = total = 0
	for _, _, path in worker_files(directories, worker_count, file_count):
		try:
			st = os.stat(path)
		except OSError:
			continue
		if stat.S_ISREG(st.st_mode):
			count += 1
			total += st.st_size
	if not count:
		return None
	return (f"Left {count} test files ({format_bytes(total)}B) in {', '.join(directories)}; "
		"use -cl/--cleanup to remove them.")

def simultaneous_worker_dir(base_directory, run_id, counter):
	return os.path.join(base_directory, f".iotest_simul_{run_id}", f"worker_{counter}")

def layout_read_worker(file_count, file_size, base_directory, run_id, counter, quiet, zeros, unique=False, block_size=None, verify=False, errors=None):
	content = FileContent(file_size, zeros, unique, new_stamp_id(), block_size, verify)
	worker_dir = simultaneous_worker_dir(base_directory, run_id, counter)
	os.makedirs(worker_dir, exist_ok=True)
	for i in range(file_count):
		i_str = str(i + 1).zfill(len(str(file_count)))
		file_name = os.path.join(worker_dir, f"read_{i_str}.bin")
		op_result = content.write(file_name, i, quiet=quiet)
		# a read file that was not laid out would be recreated inside the timed pass, so report it
		if errors is not None and op_result[0] is None:
			errors.append(str(op_result[1]))

def simultaneous_role_worker(role, file_count, file_size, base_directory, run_id, counter, results, quiet, zeros, thread_start_time, unique=False, late_starts=None, timeline=None, block_size=None, verify=False):
	local_results = []
	local_timeline = OpTimeline(counter, thread_start_time, file_count) if timeline is not None else None
	log = functools.partial(worker_color_log, int_to_color(os.getpid()), quiet=quiet)
	log(f'Worker {counter} ({role}) scheduled to start at {thread_start_time-time.perf_counter():.4f} later.')
	# readers read the layout's files and create content only if one must be recreated
	content = FileContent(file_size, zeros, unique, new_stamp_id(), block_size, verify)
	if role == 'write':
		content.prepare()
	if late_starts is not None and time.perf_counter() > thread_start_time:
		late_starts.append(time.perf_counter() - thread_start_time)
	wait_for_barrier(thread_start_time)
	worker_dir = simultaneous_worker_dir(base_directory, run_id, counter)
	os.makedirs(worker_dir, exist_ok=True)
	file_prefix = 'write_' if role == 'write' else 'read_'
	read_buffer = bytearray(block_size or file_size) if role == 'read' else None
	for i in range(file_count):
		i_str = str(i + 1).zfill(len(str(file_count)))
		file_name = os.path.join(worker_dir, f"{file_prefix}{i_str}.bin")
		write_file = functools.partial(content.write, index=i, quiet=quiet)
		attempt = time.perf_counter() if local_timeline is not None else None
		if role == 'write':
			op_result = write_file(file_name)
			if local_timeline is not None:
				local_timeline.add('write', i + 1, attempt, op_result)
			op = op_time(op_result)
			local_results.append(op)
			log(f"[Process {os.getpid()}]\tFile {i_str}/{file_count}:\t{file_name}\tWrote\t{format_op(op)}")
		else:
			op_result = read_file(file_name, None, file_size, quiet, buffer=read_buffer, recreate=write_file, block_size=block_size, verify=verify)
			if local_timeline is not None:
				local_timeline.add('read', i + 1, attempt, op_result)
			op = op_time(op_result)
			local_results.append(op)
			log(f"[Process {os.getpid()}]\tFile {i_str}/{file_count}:\t{file_name}\tRead\t{format_op(op)}")
	log(f'Worker {counter} ({role}) finished.')
	results.append(results_record(local_results))
	if local_timeline is not None:
		# one record per worker and pass, so the manager sees one append, not one per op
		timeline.append(local_timeline.record())

def cleanup_simultaneous_tree(directories, run_id):
	# the run's tree in each directory that has one
	for base in directories:
		simul_root = os.path.join(base, f".iotest_simul_{run_id}")
		if os.path.isdir(simul_root):
			shutil.rmtree(simul_root, ignore_errors=True)

def run_simultaneous_layout(file_count, file_size, directories, run_id, layout_worker_count,
		quiet, zeros, tl, unique=False, block_size=None, verify=False):
	tl.teeprint(f"Simultaneous mode: layout pass writing read files ({layout_worker_count} workers)...")
	layout_start = time.perf_counter()
	layout_processes = []
	layout_manager = Manager()
	layout_errors = layout_manager.list()
	for counter in range(layout_worker_count):
		assigned_directory = directories[counter % len(directories)]
		counter_str = worker_counter_name(counter)
		p = Process(target=layout_read_worker, args=(file_count, file_size, assigned_directory, run_id, counter_str, quiet, zeros, unique, block_size, verify, layout_errors))
		layout_processes.append(p)
	start_processes(layout_processes)
	for p in layout_processes:
		p.join()
	# one bulk fetch: iterating a proxy costs a round trip per item
	failures = layout_errors[:]
	layout_manager.shutdown()
	if failures:
		tl.teeerror(f"Layout pass could not write {len(failures)} read files, first error: {failures[0]}")
	tl.teeprint(f"Layout pass complete in {time.perf_counter() - layout_start:.4f} s")
	return failures

def normalize_directories(directory):
	if isinstance(directory, str):
		return [directory]
	if directory is None:
		return [os.getcwd()]
	return list(directory)

class RunOptions:
	# every option of one invocation; a new option is added here, with its default
	def __init__(self, *, file_size, file_count, process_count, modes, directories=None,
			quiet=True, zeros=False, stealth=False, message_end_point_address=None, no_report=False,
			threshold_to_report_anomaly=0, benchmark_time=5, unique=False, cleanup=False, json_path=None, start_at=None,
			block_size=None, verify=False, timeline=False):
		self.file_size = file_size
		self.file_count = file_count
		self.process_count = process_count
		self.modes = modes
		self.directories = normalize_directories(directories)
		self.quiet = quiet
		self.zeros = zeros
		self.stealth = stealth
		self.message_end_point_address = message_end_point_address
		self.no_report = no_report
		self.threshold_to_report_anomaly = threshold_to_report_anomaly
		self.benchmark_time = benchmark_time
		# --verify implies -u: unstamped blocks repeat (across a worker's files, and within a file
		# with -bs or -z), so the CRC could not tell a block written to the wrong place from the right one
		self.unique = unique or verify
		self.verify = verify
		self.cleanup = cleanup
		self.json_path = json_path
		self.start_at = start_at
		# as requested; run_iotest() treats one at least the file size as unset
		self.block_size = block_size
		# write <base>_ops.csv, also with no_report
		self.timeline = timeline

	def replace(self, **changes):
		values = dict(vars(self))
		values.update(changes)
		return RunOptions(**values)


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

def add_op_results(outResults, failedOps, key, samples):
	# failed ops carry their error text instead of a duration (see op_time).
	# Always add to both dicts so repeated keys get renamed in lockstep.
	times, failed = [], []
	for sample in samples:
		(failed if isinstance(sample, str) else times).append(sample)
	addToDicWithoutOverwrite(outResults, key, times)
	addToDicWithoutOverwrite(failedOps, key, failed)
	return times

def record_pass_meta(late_starts, sync_estimated, key, late, estimated):
	# kept in lockstep with outResults, so a repeated key is renamed in all of them
	addToDicWithoutOverwrite(late_starts, key, (len(late), max(late) if late else 0.0))
	addToDicWithoutOverwrite(sync_estimated, key, estimated)

def warn_late_starts(tl, mode, late, process_count):
	if late:
		tl.teelog(f"{len(late)} of {process_count} workers started late in {mode} mode, up to {max(late):.4f} s late; sync figures include the delay.", level='warning')

def start_processes(processes):
	# POSIX multiprocessing flushes stdout before starting a child, Windows does not: without this, piped
	# output there shows the workers' lines ahead of what this process printed before starting them
	sys.stdout.flush()
	sys.stderr.flush()
	for p in processes:
		p.start()

def dead_worker_errors(processes, file_count):
	# a worker hands its results over only at the very end, so one that dies
	# (e.g. it cannot create its directory) loses every op it was given
	errors = []
	for p in processes:
		if p.exitcode != 0:
			errors.extend([f"worker process exited with code {p.exitcode} before reporting results"] * file_count)
	return errors

def total_failed_ops(result):
	# layout write failures count like failed ops: they make the run exit 1
	return sum(len(errors) for errors in result.get('failedOps', {}).values()) + len(result.get('layout_errors', ()))

def benchmarkGenSpeed(file_size, file_count,zeros,results,timeout=5):
	start_time = time.perf_counter()
	c = 0
	while True:
		if zeros:
			content = b'\x00' * file_size
		else:
			content = almost_urandom(file_size)
		# free it now, or the next iteration holds two buffers while generating
		del content
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

def calc_iops(op_times, total_time, process_count):
	# call: ops per second of per-process time spent inside the op calls
	# sync: ops per second of wall-clock time for the whole pass
	op_count = len(op_times)
	total_p_time = sum(op_times) / process_count if process_count > 0 else 0
	call_iops = op_count / total_p_time if total_p_time > 0 else None
	sync_iops = op_count / total_time if total_time and total_time > 0 else None
	return call_iops, sync_iops

def format_iops(iops):
	if iops is None:
		return 'N/A'
	return f"{iops:,.2f}"

SHORT_REPORT_TIMEOUT = 10

def post_short_report(message_end_point_address, content):
	payload = json.dumps({'content': content}).encode()
	# an explicit agent: some webhook hosts (Discord) reject urllib's default Python-urllib one
	request = urllib.request.Request(message_end_point_address, data=payload, method='POST',
		headers={'Content-Type': 'application/json', 'User-Agent': f'iotest/{version}'})
	with urllib.request.urlopen(request, timeout=SHORT_REPORT_TIMEOUT):
		pass

def send_short_reports(message_end_point_address, short_reports, tl):
	# a notification failure must not cost the results, so this runs after the report is out
	if '://' not in message_end_point_address:
		# curl assumed http:// for a bare host:port/path; keep accepting those
		message_end_point_address = 'http://' + message_end_point_address
	host = urllib.parse.urlsplit(message_end_point_address).netloc or message_end_point_address
	for sent, short_report in enumerate(short_reports):
		try:
			post_short_report(message_end_point_address, short_report)
		except Exception as e:
			# the rest would most likely fail the same way, each after a full timeout
			skipped = len(short_reports) - sent - 1
			skipped_note = f" (skipping {skipped} more)" if skipped else ""
			tl.teeerror(f"Failed to send short report to {host}: {e}{skipped_note}")
			return

def files_left_before(modes, index):
	# what the passes before modes[index] leave of the plain workers' temp files: True after a write, random or
	# read pass (each writes every file it does not find), False after a comprehensive pass (it renames and
	# deletes every file), None when no pass touched them, so they are as the run found them
	for mode in reversed(modes[:index]):
		if mode in ('write', 'random', 'read'):
			return True
		if mode == 'comprehensive':
			return False
	return None

def read_pass_recreates(modes, scan, process_count):
	# {index in modes of each read pass: per worker, whether it has files to recreate in that pass}. A worker
	# that does makes its file content before the start barrier, and the memory estimate counts it
	passes = {}
	for index, mode in enumerate(modes):
		if mode == 'read':
			left = files_left_before(modes, index)
			if left is None:
				passes[index] = [wrong > 0 for _, wrong in scan[:process_count]]
			else:
				passes[index] = [not left] * process_count
	return passes

# data buffers a worker holds in a pass of each mode: its file content (writers, and the generation benchmark's
# processes), its read buffer (readers, simultaneous ones included), or both. Index and meta passes move no data
BUFFER_COPIES = {'write': 1, 'read': 1, 'random': 2, 'comprehensive': 2, 'simultaneous': 1, 'benchmark': 1,
	'index': 0, 'meta': 0}

def memory_copies(modes, read_recreates=False):
	# the most buffers a worker holds at once, in any of its passes: a read worker that recreates files also
	# holds their content
	return max((BUFFER_COPIES.get(mode, 0) + (1 if mode == 'read' and read_recreates else 0) for mode in modes),
		default=0)

# ops each process performs per file in each mode (meta is three passes of one op)
OPS_PER_FILE = {'write': 1, 'read': 1, 'random': 1, 'index': 1, 'comprehensive': 4, 'meta': 3, 'simultaneous': 1}

# memory per op performed, by mode, in the parent and the manager process together. Every run keeps each
# op's result until it ends: workers hand a pass's over as compact records (8 bytes an op), which the manager
# holds until the parent fetches them in one go, and the parent keeps them as the float lists of the result
# dict (about 32 bytes an op). --timeline adds the op's record: 21 bytes in arrays, plus the copies made
# handing a pass over. Measured as the slope of their peak RSS (VmHWM) from -fc 25000 to 75000 at -pc 8 with
# 1 KiB files (Python 3.13, Linux, fork, 3 runs per mode). Results: meta 50-63 bytes an op (the manager holds
# one of its three passes at a time), write, read, index, random and comprehensive 79-85, simultaneous
# 100-110. The timeline: 36-56 more for meta, 35-90 for the others (also counting the runs before results
# became records, as its own path did not change). Each figure is the largest slope plus about 20 %, rounded
# up to 10; write, read and index, which return the same one float per op, share the largest of theirs.
# These cover the parent and the manager only: each worker's own memory for its pass (its result list, the
# hand-over copies, and with --timeline the timeline arrays) is not counted separately, since it peaks
# before the parent decodes that pass, and the ~20 % margin covers it; account for it before tightening
# these. Failed ops' error texts (100-200 bytes each) are not counted.
RESULT_BYTES_PER_OP = {'write': 110, 'read': 110, 'index': 110, 'simultaneous': 140, 'comprehensive': 100,
	'random': 110, 'meta': 80}
TIMELINE_BYTES_PER_OP = {'write': 110, 'read': 110, 'index': 110, 'simultaneous': 100, 'comprehensive': 110,
	'random': 110, 'meta': 70}

def memory_per_process(modes, buffer_size, file_count, read_recreates=False, timeline=False):
	# a worker's data buffers, plus what the parent and the manager keep for every op it performs over the run
	op_bytes = sum(OPS_PER_FILE.get(mode, 0) * (RESULT_BYTES_PER_OP.get(mode, 0)
		+ (TIMELINE_BYTES_PER_OP.get(mode, 0) if timeline else 0)) for mode in modes)
	return buffer_size * memory_copies(modes, read_recreates) * 1.2 + file_count * op_bytes

def workers_memory(modes, buffer_size, file_count, recreates, timeline=False):
	# memory_per_process() of each worker, in counter order, by whether it recreates files in a read pass
	figures = [memory_per_process(modes, buffer_size, file_count, flag, timeline) for flag in (False, True)]
	return [figures[1 if flag else 0] for flag in recreates]

def fitting_process_count(worker_memory, available):
	# how many of the first workers fit in the available memory together, at least one
	total = 0
	for count, memory in enumerate(worker_memory):
		total += memory
		if total > available:
			return max(count, 1)
	return len(worker_memory)

def read_meminfo():
	# (MemAvailable, SwapFree) in bytes, -1 when unknown
	phyFreeMemory = -1
	swapMemory = -1
	if os.path.exists('/proc/meminfo'):
		with open('/proc/meminfo') as f:
			for line in f:
				if 'MemAvailable' in line:
					phyFreeMemory = int(line.split()[1]) * 1024
				if 'SwapFree' in line:
					swapMemory = int(line.split()[1]) * 1024
	return phyFreeMemory, swapMemory

NETWORK_FILESYSTEMS = ('nfs', 'nfs4', 'cifs', 'smb3')

def read_mountinfo():
	try:
		with open('/proc/self/mountinfo') as f:
			return f.read().splitlines()
	except OSError:
		return []

def unescape_mount_path(path):
	# mountinfo writes space, tab, newline and backslash as octal escapes (\040 ...)
	return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), path)

def find_mount(path, mountinfo_lines):
	# (fstype, source) of the deepest mount point that contains path
	best = None
	best_len = -1
	for line in mountinfo_lines:
		fields = line.split()
		if '-' not in fields:
			continue
		sep = fields.index('-')
		if sep < 5 or len(fields) < sep + 3:
			continue
		mount_point = unescape_mount_path(fields[4])
		inside = path == mount_point or path.startswith(mount_point.rstrip('/') + '/')
		# >= so a later line for the same mount point (mounted on top) wins
		if inside and len(mount_point) >= best_len:
			best = (fields[sep + 1], unescape_mount_path(fields[sep + 2]))
			best_len = len(mount_point)
	return best

def zfs_dedup_property(dataset):
	try:
		proc = subprocess.run(['zfs', 'get', '-H', '-o', 'value', 'dedup', dataset],
			stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=5)
	except Exception:
		return None
	if proc.returncode != 0:
		return None
	return proc.stdout.strip() or None

def dedup_warning(directory, mountinfo_lines):
	mount = find_mount(os.path.realpath(directory), mountinfo_lines)
	if not mount:
		return None
	fstype, source = mount
	if fstype in NETWORK_FILESYSTEMS:
		reason = f"is on {fstype} ({source}), which may deduplicate on the server"
	elif fstype == 'zfs':
		dedup = zfs_dedup_property(source)
		if not dedup or dedup == 'off':
			return None
		reason = f"is on ZFS dataset {source} with dedup={dedup}"
	else:
		return None
	return (f"Warning! {directory} {reason}. Each worker writes identical files, "
		"so results may be optimistic. Use -u/--unique_files to make every file unique.")

REPORT_SUFFIXES = ('.csv', '_report.txt', '_ops.csv')

def report_base_name(modes, directories, file_size, file_count, process_count, current_date_time):
	dir_str_repr = "_".join([d.replace('/','-').replace('\\','-').replace(':','-').replace('--','-').replace(' ','_') for d in directories])
	head = f"iotest_{'-'.join(modes)}_"
	tail = f"_fs={file_size}_fc={file_count}_pc={process_count}_{current_date_time}"
	# file names are limited to 255 bytes, so only the directory part may shrink
	room = 255 - max(len(suffix.encode('utf-8')) for suffix in REPORT_SUFFIXES) - len((head + tail).encode('utf-8'))
	if len(dir_str_repr.encode('utf-8')) > room:
		marker = '~' + hashlib.sha1(dir_str_repr.encode('utf-8')).hexdigest()[:8]
		room -= len(marker)
		kept = ''
		for char in dir_str_repr:
			if len((kept + char).encode('utf-8')) > room:
				break
			kept += char
		dir_str_repr = kept + marker
	return head + dir_str_repr + tail

def collect_timeline(timeline_passes, pass_timeline, pass_index, mode):
	if timeline_passes is not None:
		# one bulk fetch of the pass's records (one per worker), then empty the manager's list so the
		# manager process does not hold them too
		timeline_passes.append((pass_index, mode, pass_timeline[:]))
		del pass_timeline[:]

def timeline_ops(record):
	# one worker's ops of one pass, in the order it ran them, so by start: (start, op, worker, file_index, duration, error)
	counter, index_type, ops, file_indexes, starts, durations, errors = record
	file_indexes = memoryview(file_indexes).cast(index_type)
	starts = memoryview(starts).cast('d')
	durations = memoryview(durations).cast('d')
	for position, start in enumerate(starts):
		failed = position in errors
		yield (start, OP_NAMES[ops[position]], counter, file_indexes[position],
			None if failed else durations[position], errors[position] if failed else '')

def write_timeline(timeline_passes):
	def write(f):
		writer = csv.writer(f)
		writer.writerow(['pass', 'mode', 'op', 'worker', 'file_index', 'start_s', 'duration_s', 'error'])
		# passes are already in order. Within one, each worker's ops are in start order, so a lazy merge
		# orders them with one pending row per worker; equal starts keep the order the records arrived in
		for pass_index, mode, records in timeline_passes:
			for start, op, counter, file_index, duration, error in heapq.merge(*map(timeline_ops, records), key=lambda row: row[0]):
				writer.writerow([pass_index, mode, op, counter, file_index, '%.6f' % start,
					'' if duration is None else '%.6f' % duration, error])
	return write

def write_report_file(path, write_content, tl, report_errors, newline=None):
	try:
		# csv writes its own line endings, so it needs newline='' to avoid \r\r\n on Windows
		with open(path, "w", newline=newline, encoding="utf-8") as f:
			write_content(f)
	except (OSError, ValueError) as e:
		message = f"Could not write {path}: {e}"
		tl.teeerror(message)
		report_errors.append(message)
		return False
	return True

def summarize_key(times, failed, effective_pc, total_time, file_size, has_bytes):
	# raw numbers for one result key; the text report and the JSON are both built from this
	stats = {'ops': len(times), 'failed': len(failed), 'first_error': str(failed[0]) if failed else None,
		'process_count': effective_pc, 'total_bytes': file_size * len(times) if has_bytes else None,
		'bytes_per_s_call': None, 'bytes_per_s_sync': None, 'io_time_pct': None,
		'iops_call': None, 'iops_sync': None, 'latency_s': None}
	if not times:
		return stats
	# a sorted copy: the CSV keeps the arrival order
	times = sorted(times)
	count = len(times)
	stats['latency_s'] = {'avg': sum(times) / count, 'median': times[count // 2], 'p1': times[int(count * 0.01)],
		'p99': times[int(count * 0.99)], 'p99_9': times[int(count * 0.999)], 'max': times[-1]}
	stats['iops_call'], stats['iops_sync'] = calc_iops(times, total_time, effective_pc)
	if has_bytes:
		total_p_time = sum(times) / effective_pc
		if total_p_time > 0:
			stats['bytes_per_s_call'] = stats['total_bytes'] / total_p_time
		if total_time and total_time > 0:
			stats['bytes_per_s_sync'] = stats['total_bytes'] / total_time
			stats['io_time_pct'] = total_p_time / total_time * 100
	return stats

def format_key_report(mode, stats, late, estimated, zeros, dedup_notes):
	report = ["*"*80, f"Report for {mode} mode:"]
	if zeros:
		report.append("Warning! Using zeros for file content. Compressed filesystems will show optimistic results.")
	report.extend(dedup_notes)
	op_count = stats['ops'] + stats['failed']
	if stats['failed']:
		report.append(f"Failed {mode} ops:      \t{stats['failed']} of {op_count}, first error: {stats['first_error']}")
	if late[0]:
		report.append(f"Late starts:           \t{late[0]} of {stats['process_count']} workers, up to {late[1]:.4f} s late (sync figures include the delay)")
	latency = stats['latency_s']
	if latency is None:
		report.append(f"No successful {mode} ops, skipping timing statistics.")
		report.append("-"*80)
		return report
	if stats['total_bytes'] is not None:
		call_bps = stats['bytes_per_s_call']
		sync_bps = stats['bytes_per_s_sync']
		pct = stats['io_time_pct']
		report.append(f"Total {mode} size:      \t{format_bytes(stats['total_bytes'])}B")
		report.append(f"{mode} bandwidth (call):\t{format_bytes(call_bps)}B/s" if call_bps is not None else f"{mode} bandwidth (call):\tN/A")
		if call_bps is not None:
			report.append(f"                       \t{format_bytes(call_bps * 8,use_1024_bytes=False)}bps")
		report.append(f"{mode} bandwidth (sync):\t{format_bytes(sync_bps)}B/s" if sync_bps is not None else f"{mode} bandwidth (sync):\tN/A")
		if sync_bps is not None:
			report.append(f"                       \t{format_bytes(sync_bps * 8,use_1024_bytes=False)}bps")
		report.append(f"IO time % (IO wait):   \t{pct:.2f}%" if pct is not None else "IO time % (IO wait):   \tN/A")
	report.append(f"Total {mode} ops:       \t{stats['ops']}")
	report.append(f"{mode} IOPS (call):     \t{format_iops(stats['iops_call'])}")
	report.append(f"{mode} IOPS (sync):     \t{format_iops(stats['iops_sync'])}")
	report.append(f"Average {mode} time:    \t{latency['avg']:.4f} s")
	report.append(f"Median {mode} time:     \t{latency['median']:.4f} s")
	report.append(f"1 % low {mode} time:    \t{latency['p1']:.4f} s")
	report.append(f"1 % high {mode} time:   \t{latency['p99']:.4f} s")
	report.append(f"0.1 % high {mode} time: \t{latency['p99_9']:.4f} s")
	report.append(f"Highest {mode} time:    \t{latency['max']:.4f} s")
	if estimated:
		report.append("Note: sync figures are estimates: comprehensive mode splits the pass time across write/move/stat/read by their share of op time.")
	report.append("-"*80)
	return report

def build_json_summary(options, started, key_stats, late_starts, sync_estimated, report_errors, start_late=()):
	results = {}
	for key, stats in key_stats.items():
		entry = dict(stats)
		entry['sync_estimated'] = bool(sync_estimated.get(key))
		late_n, late_max = late_starts.get(key, (0, 0.0))
		entry['late_workers'] = late_n
		entry['max_late_s'] = late_max
		results[key] = entry
	return {
		'tool': 'iotest',
		'version': version,
		'host': socket.gethostname(),
		'started': started,
		'options': {'file_size': int(options.file_size), 'file_count': options.file_count,
			'process_count': options.process_count, 'directories': options.directories,
			'modes': options.modes, 'zeros': options.zeros, 'unique': options.unique,
			'start_at': options.start_at,
			'block_size': None if options.block_size is None else int(options.block_size),
			'verify': options.verify, 'timeline': options.timeline},
		# how long the synchronized first pass started after --start_at; null without it
		'start_late_s': None if options.start_at is None else (start_late[0] if start_late else 0.0),
		'results': results,
		'failed_ops': sum(stats['failed'] for stats in key_stats.values()),
		'report_errors': list(report_errors),
	}

def write_json_summary(path, summary, tl, report_errors):
	text = json.dumps(summary, indent=2) + '\n'
	if path == '-':
		sys.stdout.write(text)
		sys.stdout.flush()
	elif write_report_file(path, lambda f: f.write(text), tl, report_errors):
		tl.teeprint(f"JSON written to {path}")

def first_pass_start(start_at, process_start_delay, tl, start_notes, start_late):
	# the synchronized pass waits for start_at; if too little time is left for the
	# workers to prime, it starts after the usual delay instead and says how late
	now = time.perf_counter()
	remaining = start_at - time.time()
	start_notes.append(f"Synchronized start at {datetime.datetime.fromtimestamp(start_at).astimezone().isoformat(timespec='seconds')} (--start_at)")
	if remaining >= process_start_delay:
		start_late.append(0.0)
		tl.teeprint(f"Waiting until {datetime.datetime.fromtimestamp(start_at).astimezone().isoformat(timespec='seconds')} for --start_at ({remaining:.1f} s)")
		return now + remaining
	late_by = process_start_delay - remaining
	start_late.append(late_by)
	start_notes.append(f"Started {late_by:.1f} s late: less than the {process_start_delay:.1f} s process start delay was left")
	tl.teelog(f"Warning | iotest: --start_at is too soon; the first pass starts {late_by:.1f} s late", level='warning')
	return now + process_start_delay

def main(file_size, file_count, process_count, directory,modes,quiet,zeros,tl=None,stealth=False,message_end_point_address=None,no_report=False,threshold_to_report_anomaly = 0,benchmarkTime = 5,tuning_attempt=False,simultaneous_write_count=None,simultaneous_run_id=None,simultaneous_skip_layout=False,simultaneous_skip_cleanup=False,simultaneous_layout_workers=None,simultaneous_layout_process_count=None,unique=False,cleanup=False,json_path=None,start_at=None,block_size=None,verify=False,timeline=False):
	# the original entry point, kept for callers that pass options one by one. simultaneous_layout_process_count
	# is accepted and ignored: worker names no longer depend on the process count
	options = RunOptions(file_size=file_size, file_count=file_count, process_count=process_count,
		modes=modes, directories=directory, quiet=quiet, zeros=zeros, stealth=stealth,
		message_end_point_address=message_end_point_address, no_report=no_report,
		threshold_to_report_anomaly=threshold_to_report_anomaly, benchmark_time=benchmarkTime,
		unique=unique, cleanup=cleanup, json_path=json_path, start_at=start_at, block_size=block_size,
		verify=verify, timeline=timeline)
	return run_iotest(options, tl=tl, tuning_attempt=tuning_attempt,
		simultaneous_write_count=simultaneous_write_count, simultaneous_run_id=simultaneous_run_id,
		simultaneous_skip_layout=simultaneous_skip_layout, simultaneous_skip_cleanup=simultaneous_skip_cleanup,
		simultaneous_layout_workers=simultaneous_layout_workers)

def run_iotest(options, tl=None, tuning_attempt=False, simultaneous_write_count=None,
		simultaneous_run_id=None, simultaneous_skip_layout=False, simultaneous_skip_cleanup=False,
		simultaneous_layout_workers=None):
	# read before the tuning override below: attempts must keep the digits off the JSON's stdout too
	json_to_stdout = options.json_path == '-'
	if tuning_attempt:
		# an attempt only feeds the tuner: no report files, no short reports, no timeline
		options = options.replace(no_report=True, message_end_point_address=None, json_path=None, timeline=False)
	if options.stealth:
		options = options.replace(quiet=True, no_report=True)
	# locals: the checks below may lower process_count or zero file_count, and the caller's
	# options never change (tuning reuses them for every attempt)
	started = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
	file_size = int(options.file_size)
	file_count = options.file_count
	process_count = options.process_count
	modes = options.modes
	quiet = options.quiet
	zeros = options.zeros
	unique = options.unique
	verify = options.verify
	directories = options.directories
	block_size = effective_block_size(options.block_size, file_size)
	# what each worker holds: one block with -bs, the whole file otherwise
	buffer_size = block_size or file_size
	if not tl:
		tl = Tee_Logger.teeLogger(suppressPrintout=quiet)
	if not directories:
		raise ValueError("At least one output directory must be provided")
	for output_dir in directories:
		os.makedirs(output_dir, exist_ok=True)
	primary_directory = directories[0]
	mountinfo = read_mountinfo()
	dedup_notes = []
	# index and benchmark write no file data, so dedup cannot skew them
	if not unique and not tuning_attempt and any(mode not in ('index', 'benchmark', 'meta') for mode in modes):
		dedup_notes = [note for note in (dedup_warning(d, mountinfo) for d in directories) if note]
		for note in dedup_notes:
			tl.teelog(note, level='warning')
	processes = []
	outResults = dict()
	failedOps = {}
	layout_errors = []
	late_starts = {}
	start_notes = []
	start_late = []
	sync_estimated = {}
	totalTime = {}
	mode_process_counts = {}
	scan = None
	if any(mode in ('write', 'random', 'read') for mode in modes):
		# one os.stat per temp file, for the free-space check and for which read workers recreate files
		scan = scan_worker_files(directories, process_count, file_count, file_size)
	recreate_passes = read_pass_recreates(modes, scan, process_count)
	# per worker: whether it recreates files in some read pass, and so holds a second buffer
	recreating = [any(flags[counter] for flags in recreate_passes.values()) for counter in range(process_count)]
	worker_memory = workers_memory(modes, buffer_size, file_count, recreating, options.timeline)
	estimatedTotalMemory = sum(worker_memory)
	if modes == ['benchmark']:
		file_count = 0
	try:
		phyFreeMemory, swapMemory = read_meminfo()
	except Exception:
		phyFreeMemory, swapMemory = -1, -1
		tl.teeerror("Failed to read /proc/meminfo")
	# warn if the estimated total memory is more than 90% of the total free memory
	if phyFreeMemory + swapMemory > 0 and estimatedTotalMemory > phyFreeMemory + swapMemory:
		tl.teeerror("Estimated total memory usage is more than the all available memory to use.")
		tl.teeerror(f"Physical available memory: {format_bytes(phyFreeMemory)}B")
		tl.teeerror(f"Swap available memory: {format_bytes(swapMemory)}B")
		tl.teeerror(f"Estimated total memory usage: {format_bytes(estimatedTotalMemory)}B")
		# the first workers that fit: a smaller count runs the same first workers, on the same files
		process_count = fitting_process_count(worker_memory, phyFreeMemory + swapMemory)
		tl.teeerror(f"Reducing the number of processes to {process_count}")
	if 'simultaneous' in modes and simultaneous_write_count is not None and simultaneous_write_count > process_count - 1 >= 1:
		# the memory check may have cut process_count below the requested split
		tl.teelog(f"Reducing simultaneous writers from {simultaneous_write_count} to {process_count - 1} to keep a reader", level='warning')
		simultaneous_write_count = process_count - 1
	estimatedTotalMemory = sum(worker_memory[:process_count])
	# the disk check needs the final process_count, but runs before the 16 s memory pause
	needed, cow_extra = space_needed(modes, file_size, file_count, process_count, directories,
		simultaneous_write_count=simultaneous_write_count, layout_worker_count=simultaneous_layout_workers,
		skip_layout=simultaneous_skip_layout, mountinfo_lines=mountinfo, scan=scan)
	if not check_free_space(needed, tl, mountinfo, cow_extra=None if tuning_attempt else cow_extra):
		sys.exit(1)
	# recorded before any worker writes, so -cl after tuning also covers an attempt that
	# never finished (Ctrl+C, a failing report write, ...)
	if any(mode in ('write', 'read', 'random', 'comprehensive', 'index') for mode in modes):
		note_worker_run(process_count)
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
		# workers record a timeline only with --timeline
		timeline_passes = [] if options.timeline else None
		pass_index = 0
		is_simultaneous = 'simultaneous' in modes
		if is_simultaneous:
			if simultaneous_write_count is not None:
				gen_benchmark_workers = simultaneous_write_count
			else:
				gen_benchmark_workers = process_count // 2
		else:
			gen_benchmark_workers = process_count
		effective_benchmark_time = options.benchmark_time
		if is_simultaneous and tuning_attempt:
			effective_benchmark_time = SIMULTANEOUS_TUNING_BENCHMARK_TIME
		if is_simultaneous and gen_benchmark_workers == 0:
			genSpeed = 0
			genTimeCalc = 0
			processStartDelay = 0.005 * process_count + 1
			tl.teeprint("Skipping file generation benchmark (no write workers in simultaneous mode)")
		elif all(mode in ('read', 'index', 'meta') for mode in modes) and not any(recreating[:process_count]):
			# it estimates how long workers take to generate data before the start barrier
			genSpeed = 0
			genTimeCalc = 0
			processStartDelay = 0.005 * process_count + 1
			if 'meta' in modes:
				tl.teeprint("Skipping file generation benchmark (read, index and meta workers generate no data up front)")
			else:
				tl.teeprint("Skipping file generation benchmark (read and index workers generate no data up front)")
		else:
			if zeros:
				generator = 'zeros'
			elif numpy_available:
				generator = 'numpy SFC64'
			else:
				generator = 'random.getrandbits'
			# workers generate only the buffer they hold
			buffer_kind = 'blocks' if block_size else 'files'
			tl.teeprint(f"Benchmarking file generation performance... using {gen_benchmark_workers}x {generator} with {format_bytes(buffer_size)}B {buffer_kind} * {file_count if file_count else '∞'} for {effective_benchmark_time}s")
			for counter in range(gen_benchmark_workers):
				p = Process(target=benchmarkGenSpeed, args=(buffer_size, file_count,zeros,results,effective_benchmark_time))
				processes.append(p)
			genStartTime = time.perf_counter()
			start_processes(processes)
			for p in processes:
				p.join()
			processes.clear()
			genTime = time.perf_counter() - genStartTime
			genSpeed = sum(results[:])
			if genSpeed > 0:
				genTimeCalc = buffer_size * gen_benchmark_workers / genSpeed
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
		sync_start_at = options.start_at
		# a meta run is three passes with their own barriers; a read pass carries which workers recreate files
		pass_modes = []
		for index, mode in enumerate(modes):
			if mode == 'meta':
				pass_modes.extend(('meta_' + phase, None) for phase in META_PHASES)
			else:
				pass_modes.append((mode, recreate_passes.get(index)))
		for mode, pass_recreates in pass_modes:
			if mode == 'benchmark':
				continue
			pass_index += 1
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
				run_id = simultaneous_run_id or tl.currentDateTime
				layout_worker_count = simultaneous_layout_workers if simultaneous_layout_workers is not None else process_count
				try:
					if not simultaneous_skip_layout:
						layout_errors = run_simultaneous_layout(file_count, file_size, directories, run_id, layout_worker_count,
							quiet, zeros, tl, unique, block_size, verify)
						if layout_errors:
							# the finally below still cleans the tree up
							continue
					write_results = manager.list()
					read_results = manager.list()
					write_late = manager.list()
					read_late = manager.list()
					pass_timeline = manager.list() if timeline_passes is not None else None
					write_processes = []
					read_processes = []
					if sync_start_at is not None:
						thread_start_time = first_pass_start(sync_start_at, processStartDelay, tl, start_notes, start_late)
						sync_start_at = None
					else:
						thread_start_time = time.perf_counter() + processStartDelay
					tl.teeprint(f"Simultaneous mode: {write_count} writers + {read_count} readers...")
					for counter in range(write_count):
						assigned_directory = directories[counter % len(directories)]
						# tuning attempts at smaller counts overwrite the same writer files
						counter_str = worker_counter_name(counter)
						p = Process(target=simultaneous_role_worker, args=('write', file_count, file_size, assigned_directory, run_id, counter_str, write_results, quiet, zeros, thread_start_time, unique, write_late, pass_timeline, block_size, verify))
						write_processes.append(p)
					for counter in range(read_count):
						assigned_directory = directories[counter % len(directories)]
						# the layout pass's files: the tuner lays out for its largest process count
						# and then runs smaller attempts
						counter_str = worker_counter_name(counter)
						p = Process(target=simultaneous_role_worker, args=('read', file_count, file_size, assigned_directory, run_id, counter_str, read_results, quiet, zeros, thread_start_time, unique, read_late, pass_timeline, block_size, verify))
						read_processes.append(p)
					totalStartTime = thread_start_time
					start_processes(write_processes + read_processes)
					for p in write_processes:
						p.join()
					write_sync_time = time.perf_counter() - totalStartTime
					for p in read_processes:
						p.join()
					read_sync_time = time.perf_counter() - totalStartTime
					# one bulk fetch per list: iterating a proxy costs a round trip per item
					write_results, read_results = write_results[:], read_results[:]
					write_late, read_late = write_late[:], read_late[:]
					add_op_results(outResults, failedOps, 'write', itertools.chain(result_samples(write_results), dead_worker_errors(write_processes, file_count)))
					add_op_results(outResults, failedOps, 'read', itertools.chain(result_samples(read_results), dead_worker_errors(read_processes, file_count)))
					addToDicWithoutOverwrite(totalTime, 'write', write_sync_time)
					addToDicWithoutOverwrite(totalTime, 'read', read_sync_time)
					addToDicWithoutOverwrite(mode_process_counts, 'write', write_count)
					addToDicWithoutOverwrite(mode_process_counts, 'read', read_count)
					record_pass_meta(late_starts, sync_estimated, 'write', write_late, False)
					record_pass_meta(late_starts, sync_estimated, 'read', read_late, False)
					collect_timeline(timeline_passes, pass_timeline, pass_index, mode)
					warn_late_starts(tl, 'simultaneous', write_late + read_late, process_count)
				finally:
					if not simultaneous_skip_cleanup:
						cleanup_simultaneous_tree(directories, run_id)
						tl.teeprint("Simultaneous mode: cleaned up temporary files.")
				continue
			processes = []
			results = manager.list()
			late = manager.list()
			pass_timeline = manager.list() if timeline_passes is not None else None
			if sync_start_at is not None:
				thread_start_time = first_pass_start(sync_start_at, processStartDelay, tl, start_notes, start_late)
				sync_start_at = None
			else:
				thread_start_time = time.perf_counter() + processStartDelay
			for counter in range(process_count):
				assigned_directory = directories[counter % len(directories)]
				recreates = pass_recreates is not None and pass_recreates[counter]
				counter = worker_counter_name(counter)
				p = Process(target=worker, args=(file_count, file_size, assigned_directory, results, mode, counter, quiet, zeros, thread_start_time, unique, recreates, late, json_to_stdout, pass_timeline, block_size, verify))
				processes.append(p)
			totalStartTime = thread_start_time
			start_processes(processes)
			for p in processes:
				p.join()
			totalEndTime =  time.perf_counter() - totalStartTime
			dead_errors = dead_worker_errors(processes, file_count)
			processes.clear()
			# one bulk fetch per list: iterating a proxy costs a round trip per item
			results, late = results[:], late[:]
			collect_timeline(timeline_passes, pass_timeline, pass_index, 'meta' if mode.startswith('meta_') else mode)
			if mode == 'comprehensive':
				# a worker's record holds each file's write, move, stat and read in turn
				writes, moves, stats, reads = [add_op_results(outResults, failedOps, key,
					itertools.chain(result_samples(results, every=4, offset=offset), dead_errors))
					for offset, key in enumerate(('write', 'move', 'stat', 'read'))]
				sumWTime = sum(writes)
				sumMTime = sum(moves)
				sumSTime = sum(stats)
				sumRTime = sum(reads)
				sumAllTime = sumWTime + sumMTime + sumSTime + sumRTime
				for key in ('write','move','stat','read'):
					addToDicWithoutOverwrite(mode_process_counts,key,process_count)
					record_pass_meta(late_starts, sync_estimated, key, late, True)
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
			elif mode == 'random':
				# each op's kind is in its worker's record; dead workers never reported, so their kind is unknown
				for kind in ('write', 'read'):
					if any(OP_CODES[kind] in kinds for _, kinds, _ in results):
						add_op_results(outResults,failedOps,'random_' + kind,result_samples(results, kind=OP_CODES[kind]))
						addToDicWithoutOverwrite(totalTime,'random_' + kind,totalEndTime)
						addToDicWithoutOverwrite(mode_process_counts,'random_' + kind,process_count)
						record_pass_meta(late_starts, sync_estimated, 'random_' + kind, late, False)
				if dead_errors:
					add_op_results(outResults,failedOps,'random',dead_errors)
					addToDicWithoutOverwrite(totalTime,'random',totalEndTime)
					addToDicWithoutOverwrite(mode_process_counts,'random',process_count)
					record_pass_meta(late_starts, sync_estimated, 'random', late, False)
			else:
				add_op_results(outResults,failedOps,mode,itertools.chain(result_samples(results), dead_errors))
				addToDicWithoutOverwrite(totalTime,mode,totalEndTime)
				addToDicWithoutOverwrite(mode_process_counts,mode,process_count)
				record_pass_meta(late_starts, sync_estimated, mode, late, False)
			warn_late_starts(tl, mode, late, process_count)
	report_errors = []
	writes_reports = not options.no_report and bool(outResults)
	# an explicit --timeline is written also with -nr (and -S), like --json
	writes_timeline = timeline_passes is not None and bool(outResults)
	if writes_reports or writes_timeline:
		report_base = os.path.join(primary_directory, report_base_name(modes, directories, file_size, file_count, process_count, tl.currentDateTime))
	report = list(start_notes)
	short_reports = []
	threshold_to_report_anomaly = options.threshold_to_report_anomaly
	key_stats = {}
	for mode in outResults:
		stats = summarize_key(outResults[mode], failedOps.get(mode, []), mode_process_counts.get(mode, process_count), totalTime.get(mode), file_size, 'write' in mode or 'read' in mode)
		key_stats[mode] = stats
		failed = failedOps.get(mode, [])
		op_count = stats['ops'] + stats['failed']
		failed_note = f", {stats['failed']} of {op_count} ops failed" if failed else ''
		report.extend(format_key_report(mode, stats, late_starts.get(mode, (0, 0.0)), sync_estimated.get(mode), zeros, dedup_notes))
		if not outResults[mode]:
			if options.message_end_point_address and failed:
				short_reports.append(f"Short Report for {mode} mode: all {op_count} ops failed, first error: {failed[0]}")
			continue
		latency = stats['latency_s']
		if threshold_to_report_anomaly and threshold_to_report_anomaly > 0 and latency['p99'] > (latency['p1'] * threshold_to_report_anomaly):
			report.append("Warning | iotest: 1% high is too high compared to 1% low!")
			report.append(f"Warning | iotest: 1% high is {latency['p99']:.4f} s, 1% low is {latency['p1']:.4f} s, threshold is {threshold_to_report_anomaly}")
		if options.message_end_point_address:
			if stats['total_bytes'] is not None:
				short_report = f"Short Report for {mode} mode: Speed {format_bytes(stats['bytes_per_s_call'])}B/s, {format_iops(stats['iops_call'])} IOPS @ {stats['io_time_pct']:.2f}% IO time"
			else:
				short_report = f"Short Report for {mode} mode: {format_iops(stats['iops_call'])} IOPS, avg {latency['avg']:.4f} s, median {latency['median']:.4f} s"
			short_reports.append(short_report + failed_note)
	# the text report goes first, so a failing result file cannot lose it
	# the workers' progress digits end without a newline: end their line on the stream they went to,
	# not inside the report, which the logger may prefix ("[INFO] ") on the digits' line
	if quiet:
		print(file=sys.stderr if json_to_stdout else None)
	tl.teeprint('\n'.join(report))
	if writes_reports:
		csv_file_name = report_base + REPORT_SUFFIXES[0]
		def write_csv(f):
			# the headers are just the keys in the dic, the values are the list of results
			f.write(','.join([key+'_time' for key in outResults.keys()]) + '\n')
			# columns can differ in length (uneven simultaneous splits, failed ops), so pad short ones with blank cells
			for row in itertools.zip_longest(*outResults.values(), fillvalue=''):
				f.write(','.join(map(str,row)) + '\n')
		if write_report_file(csv_file_name, write_csv, tl, report_errors):
			tl.teeprint(f"\nResults written to {csv_file_name}")
		report_file_name = report_base + REPORT_SUFFIXES[1]
		if write_report_file(report_file_name, lambda f: f.write('\n'.join(report)), tl, report_errors):
			tl.teeprint(f"Report written to {report_file_name}")
	if writes_timeline:
		ops_file_name = report_base + REPORT_SUFFIXES[2]
		if write_report_file(ops_file_name, write_timeline(timeline_passes), tl, report_errors, newline=''):
			tl.teeprint(f"Timeline written to {ops_file_name}")
	if writes_reports:
		print(f"Log file: {tl.logFileName}")
	if options.json_path:
		write_json_summary(options.json_path, build_json_summary(options, started, key_stats, late_starts, sync_estimated, report_errors, start_late), tl, report_errors)
	if options.cleanup:
		cleanup_worker_files(directories, process_count, file_count)
		if 'meta' in modes:
			cleanup_meta_dirs(directories, process_count)
	if options.message_end_point_address and short_reports:
		send_short_reports(options.message_end_point_address, short_reports, tl)
	return {
		'outResults': outResults,
		'totalTime': totalTime,
		'process_count': process_count,
		'file_size': file_size,
		'file_count': file_count,
		'mode_process_counts': mode_process_counts,
		'failedOps': failedOps,
		'late_starts': late_starts,
		'sync_estimated': sync_estimated,
		'report_errors': report_errors,
		'layout_errors': layout_errors,
	}

MODE_ALIASES = {
	'r': ['read'],
	'w': ['write'],
	'rw': ['write', 'read'],
	'wr': ['write', 'read'],
	'i': ['index'],
	'm': ['meta'],
	'rwi': ['write', 'read', 'index'],
	'wri': ['write', 'read', 'index'],
	'c': ['comprehensive'],
	'b': ['benchmark'],
	'read': ['read'],
	'write': ['write'],
	'random': ['random'],
	'index': ['index'],
	'meta': ['meta'],
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

def _simultaneous_session_kwargs(run_id, skip_layout=True, skip_cleanup=True):
	return {
		'simultaneous_run_id': run_id,
		'simultaneous_skip_layout': skip_layout,
		'simultaneous_skip_cleanup': skip_cleanup,
	}

def _run_tuning_main(options, tl, process_count, simultaneous_write_count=None, simultaneous_session=None):
	kwargs = simultaneous_session or {}
	# tuning_attempt=True: no report files and no short reports
	result = run_iotest(options.replace(process_count=process_count), tl=tl, tuning_attempt=True,
		simultaneous_write_count=simultaneous_write_count, **kwargs)
	failed = total_failed_ops(result)
	if failed:
		tl.teeerror(f"Tuning attempt had {failed} failed ops; not tuning against incomplete results.")
		sys.exit(1)
	return result

def _finalize_tuning_run(options, tl, process_count, simultaneous_write_count=None, simultaneous_session=None):
	if not options.no_report or options.message_end_point_address or options.json_path or options.timeline:
		kwargs = simultaneous_session or {}
		result = run_iotest(options.replace(process_count=process_count), tl=tl,
			simultaneous_write_count=simultaneous_write_count, **kwargs)
		if total_failed_ops(result) or result['report_errors']:
			sys.exit(1)

def tune_single_role(options, tl, role, target_bps):
	pc = options.process_count
	tl.teeprint(f"Target throughput tuning: {role} target={format_bytes(target_bps)}B/s, starting process_count={pc}")
	for attempt in range(1, MAX_TUNING_ATTEMPTS + 1):
		result = _run_tuning_main(options, tl, pc)
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
			_finalize_tuning_run(options, tl, effective_pc)
			return 0
		if effective_pc <= 1:
			tl.teeprint(f"At process_count=1, {role} measured {format_bytes(measured)}B/s still above target; "
				f"accepting as best effort.")
			_finalize_tuning_run(options, tl, 1)
			return 0
		new_pc = max(1, int(effective_pc * target_bps / measured))
		if new_pc >= effective_pc:
			new_pc = effective_pc - 1
		pc = new_pc
	tl.teeerror("Target throughput tuning exceeded maximum attempts.")
	sys.exit(1)

def tune_dual_role_shared_pc(options, tl, write_target, read_target):
	pc = options.process_count
	tl.teeprint(f"Target throughput tuning: write target={format_bytes(write_target)}B/s, "
		f"read target={format_bytes(read_target)}B/s, starting process_count={pc}")
	for attempt in range(1, MAX_TUNING_ATTEMPTS + 1):
		result = _run_tuning_main(options, tl, pc)
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
			_finalize_tuning_run(options, tl, effective_pc)
			return 0
		if effective_pc <= 1:
			tl.teeprint("At process_count=1, both write and read still above target; accepting as best effort.")
			_finalize_tuning_run(options, tl, 1)
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

def tune_simultaneous(options, tl, write_target, read_target):
	options = options.replace(benchmark_time=SIMULTANEOUS_TUNING_BENCHMARK_TIME)
	max_pc = options.process_count
	pc = options.process_count
	directories = options.directories
	for output_dir in directories:
		os.makedirs(output_dir, exist_ok=True)
	run_id = tl.currentDateTime
	layout_workers = max(max_pc - 1, 1)
	sim_session = _simultaneous_session_kwargs(run_id)
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
		_finalize_tuning_run(options, tl, pc, write_count, simultaneous_session=sim_session)
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
	# the layout is written here, outside run_iotest(), so check before it; the split
	# can move to max_pc - 1 writers while tuning
	needed, _ = space_needed(['simultaneous'], options.file_size, options.file_count, max_pc, directories,
		simultaneous_write_count=max_pc - 1, layout_worker_count=layout_workers)
	if not check_free_space(needed, tl):
		sys.exit(1)
	try:
		layout_failures = run_simultaneous_layout(options.file_count, options.file_size, directories, run_id, layout_workers,
			options.quiet, options.zeros, tl, options.unique, effective_block_size(options.block_size, options.file_size),
			options.verify)
		if layout_failures:
			sys.exit(1)
		for attempt in range(1, MAX_TUNING_ATTEMPTS + 1):
			result = _run_tuning_main(options, tl, pc, write_count, simultaneous_session=sim_session)
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
		cleanup_simultaneous_tree(directories, run_id)
		tl.teeprint("Simultaneous mode: cleaned up temporary files.")

def tune_for_targets(options, tl, write_target, read_target):
	global largest_worker_run
	if 'simultaneous' not in options.modes:
		# once up front: attempts check space quietly; tune_simultaneous checks its own layout
		for output_dir in options.directories:
			os.makedirs(output_dir, exist_ok=True)
		needed, cow_extra = space_needed(options.modes, options.file_size, options.file_count,
			options.process_count, options.directories)
		if not check_free_space(needed, tl, cow_extra=cow_extra):
			sys.exit(1)
	# clean up once at the end, not in each run: deleting files between attempts would
	# make the next read attempt recreate them inside its timed pass
	largest_worker_run = 0
	try:
		_tune_for_targets(options.replace(cleanup=False), tl, write_target, read_target)
	finally:
		if options.cleanup:
			cleanup_worker_files(options.directories, largest_worker_run, options.file_count)

def _tune_for_targets(options, tl, write_target, read_target):
	modes = options.modes
	if 'simultaneous' in modes:
		tune_simultaneous(options, tl, write_target, read_target)
		return
	is_comprehensive = 'comprehensive' in modes
	is_dual_phase = 'write' in modes and 'read' in modes and not is_comprehensive
	if write_target and read_target and (is_comprehensive or is_dual_phase):
		tune_dual_role_shared_pc(options, tl, write_target, read_target)
		return
	if write_target:
		tune_single_role(options, tl, 'write', write_target)
		return
	if read_target:
		tune_single_role(options, tl, 'read', read_target)

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

def start_at_arg(value):
	try:
		number = float(value)
	except ValueError:
		raise argparse.ArgumentTypeError(f"invalid epoch seconds: {value!r}")
	if not math.isfinite(number) or number <= time.time():
		raise argparse.ArgumentTypeError(f"must be a time in the future (epoch seconds), got {value}")
	return number

def positive_int(value):
	number = int(value)
	if number < 1:
		raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
	return number

def size_arg(value):
	try:
		number = parse_size_arg(value, unitless_as_mb=True)
	except (ValueError, OverflowError):
		raise argparse.ArgumentTypeError(f"invalid size value: {value!r}")
	if number < 1:
		raise argparse.ArgumentTypeError(f"must be at least 1 byte, got {value}")
	return number

def climain():
	default_file_size_str = '30'
	default_file_count = 50
	default_process_count = 36

	parser = argparse.ArgumentParser(description="Test total disk bandwidth. Default mode is write (w). Comprehensive mode: write -> move -> stat -> read")
	parser.add_argument("-fs","--file_size", type=size_arg, help="File size (default:30), defaults to mb, can specify in t(b),g(b),m(b),k(b),b", default=None)
	parser.add_argument("-bs","--block_size", type=size_arg, help="Write and read each file in calls of this size from one buffer of this size per worker, instead of the whole file in one call (default). Same units as -fs", default=None)
	parser.add_argument("-fc","--file_count", type=positive_int, help="Number of files to create and read per process (default:50)",default=None)
	parser.add_argument("-t",'-pc',"--process_count", type=positive_int, help="Number of processes to run concurrently (default:36)", default=None)
	parser.add_argument("-ts","--total_size", type=size_arg, help="Total size target. If -fs/-fc/-pc is omitted, auto-calculate one missing arg to fit this total size.")
	parser.add_argument('-d',"--directory", action='append', type=str, help="Directory to put the files in. Repeat -d to round-robin workers across directories (default:<pwd>)", default=None)
	parser.add_argument('-ld',"--log_directory", type=str, help="Directory to put the log files in (default:/var/log/)", default='/var/log/')
	parser.add_argument("mode", nargs='*', type=str, help="""The mode the script will operate in (default: write / w).
 COMPREHENSIVE: per-thread write -> move -> stat -> read (what your code sees).
 WRITE: batched all-thread write.
 READ: batched all-thread read.
 INDEX: creates --file_count index folders, stat each, then delete.
 META / M: create, stat, then unlink --file_count empty files per process, each as its own timed pass (keys meta_create, meta_stat, meta_unlink); -fs does not apply.
 RWI / WRI: run write, then read, then index sequentially in batch mode.
 SIMULTANEOUS / S: layout write pass for read files, then concurrent write+read on separate files; cleans up afterward.""",choices=['read', 'write','random','index','meta','m','benchmark','comprehensive','simultaneous','r','w','rw','wr','i','rwi','wri','c','b','s'], default="w")
	parser.add_argument("-q","--quiet", action="store_true", help="Kept for compatibility: quiet is the default. Use -v for per-file output",default=True)
	parser.add_argument("-v","--verbose", action="store_true", help="Verbose output",default=False)
	parser.add_argument("-S",'--stealth', action="store_true", help="Suppress verbose output and verbose log file",default=False)
	parser.add_argument('-nl',"--no_log", action="store_true", help="Do not write log files",default=False)
	parser.add_argument('-nr',"--no_report", action="store_true", help="Do not write report files",default=False)
	parser.add_argument('-tl',"--timeline", action="store_true", help="Also write <report name>_ops.csv, the start time, duration and error of every operation, even with -nr or -S. Costs up to about %d bytes of memory per operation" % max(TIMELINE_BYTES_PER_OP.values()),default=False)
	parser.add_argument("-z","--zeros", action="store_true", help="Use zeros instead of random numbers. Use this if you are sure no write compression is available. Potentially higher write accuracy.",default=False)
	parser.add_argument("-u","--unique_files", action="store_true", help="Stamp every 512-byte block with (file, block, worker) so no two blocks of the run are identical. Use on storage that deduplicates (iotest warns when it detects some). Adds a few ms per 30 MiB file to sync times.",default=False)
	parser.add_argument("--verify", action="store_true", help="End every file with a CRC-32 of its other bytes and check it on every read, in this run or a later one, to detect bit flips and torn or misplaced writes. Implies -u. Needs a file size of at least 28 bytes; files written without --verify fail the check",default=False)
	parser.add_argument("-j","--json", dest="json_path", metavar="PATH", type=str, help="Write a JSON summary of the run to PATH after the text report. Use - for stdout (with -S, stdout then carries only the JSON)",default=None)
	parser.add_argument("-cl","--cleanup", action="store_true", help="Delete the test files when the run ends (default: keep them, so a later read run can reuse them)",default=False)
	parser.add_argument('-addr',"--message_end_point_address", type=str, help="The end point address of the message")
	parser.add_argument('--threshold_to_report_anomaly', type=float, help="The threshold to report if 1 percent high is higher then 1 percent low * <threshold_to_report_anomaly>",default=0)
	parser.add_argument('-bt',"--benchmark_time", type=int, help="The time in seconds to run the benchmark for file generation speed",default=5)
	parser.add_argument("-wtpt", "--write_target_throughput", type=size_arg,
		help="Target sync write throughput (e.g. 500m, 2g). Auto-tunes the first write pass.")
	parser.add_argument("-rtpt", "--read_target_throughput", type=size_arg,
		help="Target sync read throughput (e.g. 500m, 2g). Auto-tunes the first read pass.")
	parser.add_argument("--start_at", type=start_at_arg, metavar="EPOCH", default=None,
		help="Start the first timed pass at this time, in seconds since the Unix epoch (e.g. $(date -d '+1 min' +%%s)), so several hosts start together. Clocks must be in sync")
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

	args.file_size = args.file_size if file_size_specified else parse_size_arg(default_file_size_str)
	args.file_count = args.file_count if file_count_specified else default_file_count
	args.process_count = args.process_count if process_count_specified else default_process_count

	if args.total_size is not None:
		total_size = args.total_size
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
	# after -ts, which can set the file size
	if args.verify and args.file_size < 28:
		parser.error(f"--verify requires a file size of at least 28 bytes, got {args.file_size}")

	modes = parse_modes(args.mode)
	write_target = None
	read_target = None
	if args.write_target_throughput is not None:
		if not supports_write_throughput_tuning(modes):
			parser.error("-wtpt/--write_target_throughput requires a mode with a write pass (write, rw, comprehensive, simultaneous, etc.)")
		write_target = args.write_target_throughput
	if args.read_target_throughput is not None:
		if not supports_read_throughput_tuning(modes):
			parser.error("-rtpt/--read_target_throughput requires a mode with a read pass (read, rw, comprehensive, simultaneous, etc.)")
		read_target = args.read_target_throughput
	throughput_tuning = write_target is not None or read_target is not None
	if throughput_tuning and args.start_at is not None:
		parser.error("--start_at cannot be combined with throughput tuning")
	# without -d, args.directory is None and RunOptions uses the current directory
	options = RunOptions(file_size=args.file_size, file_count=args.file_count, process_count=args.process_count,
		modes=modes, directories=args.directory, quiet=args.quiet, zeros=args.zeros, stealth=args.stealth,
		message_end_point_address=args.message_end_point_address, no_report=args.no_report,
		threshold_to_report_anomaly=args.threshold_to_report_anomaly, benchmark_time=args.benchmark_time,
		unique=args.unique_files, cleanup=args.cleanup, json_path=args.json_path, start_at=args.start_at,
		block_size=args.block_size, verify=args.verify, timeline=args.timeline)
	# -S keeps the run settings out of the output, in the log only
	echo = tl.info if args.stealth else tl.teeprint
	echo(f'Running in {modes} modes...')
	echo(f'File size: {format_bytes(options.file_size)}B')
	if options.block_size is not None:
		echo(f'Block size: {format_bytes(options.block_size)}B')
	echo(f'Number of files: {options.file_count}')
	echo(f'Number of processes: {options.process_count}')
	if write_target is not None:
		echo(f'Write target throughput: {format_bytes(write_target)}B/s')
	if read_target is not None:
		echo(f'Read target throughput: {format_bytes(read_target)}B/s')
	if 'benchmark' not in modes and len(modes) == 1:
		echo(f'Writing to {", ".join(options.directories)}')
	if throughput_tuning:
		tune_for_targets(options, tl, write_target, read_target)
		result = None
	else:
		result = run_iotest(options, tl=tl)
	if not options.cleanup and ('write' in modes or 'random' in modes or 'read' in modes):
		hint = leftover_hint(options.directories, largest_worker_run, options.file_count)
		if hint:
			tl.teeprint(hint)
	if result is not None and (total_failed_ops(result) or result['report_errors']):
		sys.exit(1)
	
if __name__ == "__main__":
	climain()