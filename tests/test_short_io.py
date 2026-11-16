import os

import pytest

# Linux caps a single read()/write() at 2 GiB - 4 KiB; these fakes reproduce
# that partial-transfer behaviour at a size a test can afford.
REAL_WRITE = os.write
REAL_READ = os.read
SHORT = 1000


def short_writev(fd, buffers):
    return REAL_WRITE(fd, bytes(buffers[0][:SHORT]))


def short_readv(fd, buffers):
    buf = buffers[0]
    data = REAL_READ(fd, min(len(buf), SHORT))
    buf[:len(data)] = data
    return len(data)


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_create_file_finishes_partial_writes(iotest_mod, tmp_path, monkeypatch):
    content = os.urandom(2500)
    target = tmp_path / "f.bin"
    monkeypatch.setattr(iotest_mod.os, "writev", short_writev)
    op = iotest_mod.op_time(iotest_mod.create_file(str(target), content, len(content), quiet=True))
    assert not isinstance(op, str), op
    assert target.read_bytes() == content


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
@pytest.mark.parametrize("op", ["write", "read"])
def test_a_refused_call_fails_the_op_on_its_first_error(iotest_mod, tmp_path, monkeypatch, op):
    # no retry: the op fails with the error the OS gave, after one call
    target = tmp_path / "f.bin"
    target.write_bytes(b"\x01" * 64)
    calls = []

    def refused(fd, buffers):
        calls.append(len(buffers[0]))
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(iotest_mod.os, op + "v", refused)
    if op == "write":
        result = iotest_mod.create_file(str(target), b"\x01" * 64, 64, quiet=True)
    else:
        result = iotest_mod.read_file(str(target), 64, quiet=True)
    assert iotest_mod.op_time(result) == "OSError: [Errno 5] Input/output error"
    assert calls == [64]


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_create_file_no_progress_is_a_failure(iotest_mod, tmp_path, monkeypatch):
    target = tmp_path / "f.bin"
    monkeypatch.setattr(iotest_mod.os, "writev", lambda fd, buffers: 0)
    op = iotest_mod.op_time(iotest_mod.create_file(str(target), b"\x01" * 64, 64, quiet=True))
    assert isinstance(op, str)
    assert str(target) in op


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_read_file_finishes_partial_reads(iotest_mod, tmp_path, monkeypatch):
    content = os.urandom(2500)
    target = tmp_path / "f.bin"
    target.write_bytes(content)
    consumed = []

    def counting_short_readv(fd, buffers):
        n = short_readv(fd, buffers)
        consumed.append(n)
        return n

    monkeypatch.setattr(iotest_mod.os, "readv", counting_short_readv)
    op = iotest_mod.op_time(iotest_mod.read_file(str(target), len(content), quiet=True))
    assert not isinstance(op, str), op
    assert sum(consumed) == 2500


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_read_file_early_eof_is_a_failure(iotest_mod, tmp_path, monkeypatch):
    target = tmp_path / "f.bin"
    target.write_bytes(b"\x01" * 64)
    monkeypatch.setattr(iotest_mod.os, "readv", lambda fd, buffers: 0)
    op = iotest_mod.op_time(iotest_mod.read_file(str(target), 64, quiet=True))
    assert isinstance(op, str)
    assert str(target) in op


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
def test_each_syscall_is_capped_at_max_io_chunk(iotest_mod, tmp_path, monkeypatch):
    # macOS rejects a single read/write over INT_MAX with EINVAL instead of
    # transferring part of it, so iotest must never ask for more than the cap
    content = os.urandom(2500)
    target = tmp_path / "f.bin"
    requested = []
    real_writev, real_readv = os.writev, os.readv

    def spy_writev(fd, buffers):
        requested.append(len(buffers[0]))
        return real_writev(fd, buffers)

    def spy_readv(fd, buffers):
        requested.append(len(buffers[0]))
        return real_readv(fd, buffers)

    monkeypatch.setattr(iotest_mod, "MAX_IO_CHUNK", 1024)
    monkeypatch.setattr(iotest_mod.os, "writev", spy_writev)
    monkeypatch.setattr(iotest_mod.os, "readv", spy_readv)
    iotest_mod.create_file(str(target), content, len(content), quiet=True)
    iotest_mod.read_file(str(target), len(content), quiet=True)
    assert target.read_bytes() == content
    assert requested == [1024, 1024, 452, 1024, 1024, 452]


@pytest.mark.skipif(not hasattr(os, "writev"), reason="os.writev and os.readv are not available on Windows")
@pytest.mark.parametrize("call,op", [("posix_fadvise", "write"), ("writev", "write"), ("readv", "read")])
def test_ctrl_c_during_io_is_not_swallowed(iotest_mod, tmp_path, monkeypatch, call, op):
    # the file helpers turn an OSError into a failed op, never an interrupt
    if not hasattr(os, call):
        pytest.skip(f"no os.{call} here")
    target = tmp_path / "f.bin"
    target.write_bytes(b"\x01" * 64)

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(iotest_mod.os, call, interrupted)
    with pytest.raises(KeyboardInterrupt):
        if op == "write":
            iotest_mod.create_file(str(target), b"\x01" * 64, 64, quiet=True)
        else:
            iotest_mod.read_file(str(target), 64, quiet=True)
