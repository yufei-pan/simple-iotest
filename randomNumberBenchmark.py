#!/usr/bin/env python3
import timeit

setup = '''
import numpy as np, math, random
from numpy.random import (
    Generator,
    PCG64,
    PCG64DXSM,
    MT19937,
    Philox,
    SFC64,
)

# RNGs
rng = Generator(PCG64())           # default PCG64
rng_dxsm = Generator(PCG64DXSM())  # DXSM variant
rng_mt = Generator(MT19937())      # Mersenne Twister
rng_ph = Generator(Philox())       # Philox
rng_sf = Generator(SFC64())        # SFC64

def almost_urandom(n):
    try:
        return random.getrandbits(8 * n).to_bytes(n, 'big')
    except OverflowError:
        return almost_urandom(n // 2) + almost_urandom(n - n // 2)

def v1(n):
    return np.random.bytes(n)

def v2(n):
    n32 = math.ceil(n/4)
    u32 = rng.integers(0, 2**32, size=n32, dtype='uint32')
    return u32.view('uint8')[:n].tobytes()

def v3(n):
    n_u64 = (n + 7) // 8
    u64 = rng.bit_generator.random_raw(n_u64)
    return u64.view('uint8')[:n].tobytes()

def v4(n):
    a = rng.integers(0, 256, size=n, dtype=np.uint8)
    return a.tobytes()

def v5(n):
    n_u64 = (n + 7) // 8
    u64 = rng_dxsm.bit_generator.random_raw(n_u64)
    return u64.view('uint8')[:n].tobytes()

def v6(n):
    n_u64 = (n + 7) // 8
    u64 = rng_mt.bit_generator.random_raw(n_u64)
    return u64.view('uint8')[:n].tobytes()

def v7(n):
    n_u64 = (n + 7) // 8
    u64 = rng_ph.bit_generator.random_raw(n_u64)
    return u64.view('uint8')[:n].tobytes()

def v8(n):
    n_u64 = (n + 7) // 8
    u64 = rng_sf.bit_generator.random_raw(n_u64)
    return u64.view('uint8')[:n].tobytes()
'''

thirty_mb = 30 * 1024 * 1024
number = 20

print("almost_urandom:        ",
      timeit.timeit(f'almost_urandom({thirty_mb})', setup=setup, number=number))
print("v1 (np.random.bytes):   ",
      timeit.timeit(f'v1({thirty_mb})', setup=setup, number=number))
print("v2 (integers→U32):      ",
      timeit.timeit(f'v2({thirty_mb})', setup=setup, number=number))
print("v3 (PCG64 raw):         ",
      timeit.timeit(f'v3({thirty_mb})', setup=setup, number=number))
print("v4 (integers→U8):       ",
      timeit.timeit(f'v4({thirty_mb})', setup=setup, number=number))
print("v5 (PCG64DXSM raw):     ",
      timeit.timeit(f'v5({thirty_mb})', setup=setup, number=number))
print("v6 (MT19937 raw):       ",
      timeit.timeit(f'v6({thirty_mb})', setup=setup, number=number))
print("v7 (Philox raw):        ",
      timeit.timeit(f'v7({thirty_mb})', setup=setup, number=number))
print("v8 (SFC64 raw):         ",
      timeit.timeit(f'v8({thirty_mb})', setup=setup, number=number))