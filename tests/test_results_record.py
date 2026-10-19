import itertools
import math
import random


def samples_of(iotest_mod, records, **kwargs):
    return list(iotest_mod.result_samples(records, **kwargs))


def test_record_round_trips_durations_and_failures(iotest_mod):
    rng = random.Random(1)
    samples = [rng.random() / 100 for _ in range(50)] + [0.0, 1e-9, "OSError: [Errno 28] No space left on device"]
    rng.shuffle(samples)
    record = iotest_mod.results_record(samples)
    durations, kinds, errors = record
    # 8 bytes an op, a NaN in a failed op's place, its error text on the side
    assert type(durations) is bytes and len(durations) == 8 * len(samples)
    assert kinds is None
    (position,) = errors
    assert errors[position] == "OSError: [Errno 28] No space left on device"
    assert math.isnan(memoryview(durations).cast("d")[position])
    # the exact floats come back, so the csv's str() of them is unchanged
    assert repr(samples_of(iotest_mod, [record])) == repr(samples)


def test_records_come_back_in_arrival_order_and_may_be_empty(iotest_mod):
    first = iotest_mod.results_record([0.5, "boom"])
    empty = iotest_mod.results_record([])
    last = iotest_mod.results_record([0.25])
    assert empty == (b"", None, {})
    assert samples_of(iotest_mod, [first, empty, last]) == [0.5, "boom", 0.25]
    assert samples_of(iotest_mod, [empty]) == []


def test_random_records_keep_each_ops_kind(iotest_mod):
    write, read = iotest_mod.OP_CODES["write"], iotest_mod.OP_CODES["read"]
    a = iotest_mod.results_record([0.5, "read failed", 0.125], [write, read, write])
    b = iotest_mod.results_record([0.25], [read])
    assert a[1] == bytes([write, read, write])
    assert samples_of(iotest_mod, [a, b], kind=write) == [0.5, 0.125]
    assert samples_of(iotest_mod, [a, b], kind=read) == ["read failed", 0.25]
    only_writes = iotest_mod.results_record([0.5], [write])
    assert not any(read in kinds for _, kinds, _ in [only_writes])


def test_comprehensive_records_hold_four_ops_per_file(iotest_mod):
    # two files of write, move, stat, read; the second file's stat failed
    record = iotest_mod.results_record([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, "stat failed", 8.0])
    other = iotest_mod.results_record([9.0, 10.0, 11.0, 12.0])
    per_key = [samples_of(iotest_mod, [record, other], every=4, offset=offset) for offset in range(4)]
    assert per_key == [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, "stat failed", 11.0], [4.0, 8.0, 12.0]]


def test_add_op_results_takes_the_samples_in_one_pass(iotest_mod):
    out, failed = {}, {}
    record = iotest_mod.results_record([0.5, "boom", 0.25])
    times = iotest_mod.add_op_results(out, failed, "write", itertools.chain(iotest_mod.result_samples([record]), ["dead"]))
    assert times == [0.5, 0.25]
    assert out == {"write": [0.5, 0.25]} and failed == {"write": ["boom", "dead"]}


def test_worker_hands_over_one_results_record(iotest_mod, work_dir):
    results = []
    iotest_mod.worker(3, 1024, str(work_dir), results, "random", "0", True, True, 0)
    (record,) = results
    durations, kinds, errors = record
    assert len(durations) == 3 * 8 and len(kinds) == 3 and errors == {}
    assert set(kinds) <= {iotest_mod.OP_CODES["write"], iotest_mod.OP_CODES["read"]}
    results = []
    iotest_mod.worker(2, 1024, str(work_dir), results, "comprehensive", "0", True, True, 0)
    ((durations, kinds, errors),) = results
    assert len(durations) == 2 * 4 * 8 and kinds is None and errors == {}


def test_worker_record_carries_a_failed_ops_error(iotest_mod, work_dir):
    # a directory where the second file goes makes its write fail
    (work_dir / "0").mkdir()
    (work_dir / "0" / "temp_2.bin").mkdir()
    results = []
    iotest_mod.worker(2, 1024, str(work_dir), results, "write", "0", True, True, 0)
    ((durations, kinds, errors),) = results
    assert list(errors) == [1]
    samples = samples_of(iotest_mod, results)
    assert isinstance(samples[0], float) and samples[1] == errors[1]
