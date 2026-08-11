def test_version_string(iotest_mod):
    assert isinstance(iotest_mod.version, str)
    assert iotest_mod.version == iotest_mod.__version__
