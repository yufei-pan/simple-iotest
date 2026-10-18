import os
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="parses Linux /proc/self/mountinfo with POSIX paths")


def mountinfo_line(mount_point, fstype, source):
    # same layout as /proc/self/mountinfo: fields, "-" separator, fstype, source, super options
    return f"36 1 0:50 / {mount_point} rw,relatime shared:1 - {fstype} {source} rw"


ROOT = mountinfo_line("/", "ext4", "/dev/sda1")


@pytest.mark.parametrize("fstype", ["nfs", "nfs4", "cifs", "smb3"])
def test_network_filesystems_warn(iotest_mod, fstype):
    lines = [ROOT, mountinfo_line("/mnt/nas", fstype, "server:/export")]
    warning = iotest_mod.dedup_warning("/mnt/nas/run1", lines)
    assert warning is not None
    assert fstype in warning
    assert "/mnt/nas/run1" in warning


def test_local_filesystem_is_quiet(iotest_mod):
    lines = [ROOT, mountinfo_line("/data", "xfs", "/dev/nvme0n1p1")]
    assert iotest_mod.dedup_warning("/data/run1", lines) is None


def test_longest_mount_point_wins(iotest_mod):
    lines = [ROOT, mountinfo_line("/mnt", "nfs4", "server:/export"),
             mountinfo_line("/mnt/local", "xfs", "/dev/sdb1")]
    assert iotest_mod.dedup_warning("/mnt/local/run1", lines) is None
    assert iotest_mod.dedup_warning("/mnt/other", lines) is not None


def test_mount_point_matches_whole_path_components(iotest_mod):
    # /mnt/nasty is not inside the /mnt/nas mount
    lines = [ROOT, mountinfo_line("/mnt/nas", "nfs4", "server:/export")]
    assert iotest_mod.dedup_warning("/mnt/nasty", lines) is None


def test_escaped_mount_point(iotest_mod):
    # mountinfo writes a space in a mount point as \040
    lines = [ROOT, mountinfo_line("/mnt/my\\040nas", "nfs4", "server:/export")]
    assert iotest_mod.dedup_warning("/mnt/my nas/run1", lines) is not None


@pytest.mark.parametrize("dedup,warns", [("on", True), ("sha256,verify", True), ("off", False), (None, False)])
def test_zfs_warns_only_when_dedup_is_on(iotest_mod, monkeypatch, dedup, warns):
    asked = []

    def fake_zfs_dedup(dataset):
        asked.append(dataset)
        return dedup

    monkeypatch.setattr(iotest_mod, "zfs_dedup_property", fake_zfs_dedup)
    lines = [ROOT, mountinfo_line("/tank/data", "zfs", "tank/data")]
    warning = iotest_mod.dedup_warning("/tank/data/run1", lines)
    assert asked == ["tank/data"]
    assert (warning is not None) == warns


def test_main_warns_and_reports_on_dedup_storage(run_main, work_dir, rec_tl, iotest_mod, monkeypatch):
    lines = [ROOT, mountinfo_line(os.path.realpath(work_dir), "nfs4", "server:/export")]
    monkeypatch.setattr(iotest_mod, "read_mountinfo", lambda: lines)
    run_main(["write"], tl=rec_tl)
    report = "\n".join(rec_tl.printed)
    assert "nfs4" in report[report.index("Report for write mode"):]


def test_main_skips_the_warning_with_unique(run_main, work_dir, rec_tl, iotest_mod, monkeypatch):
    lines = [ROOT, mountinfo_line(os.path.realpath(work_dir), "nfs4", "server:/export")]
    monkeypatch.setattr(iotest_mod, "read_mountinfo", lambda: lines)
    run_main(["write"], tl=rec_tl, unique=True)
    assert not any("nfs4" in m for m in rec_tl.messages)


def test_modes_that_write_nothing_skip_dedup_detection(run_main, work_dir, rec_tl, iotest_mod, monkeypatch):
    lines = [ROOT, mountinfo_line(os.path.realpath(work_dir), "nfs4", "server:/export")]
    monkeypatch.setattr(iotest_mod, "read_mountinfo", lambda: lines)
    run_main(["index"], tl=rec_tl)
    assert not any("nfs4" in m for m in rec_tl.messages)
