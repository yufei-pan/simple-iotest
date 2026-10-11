import csv


def test_csv_keeps_every_sample_when_columns_differ(run_main, work_dir, rec_tl):
    # simultaneous with 3 processes = 1 writer + 2 readers -> 2 write samples, 4 read samples
    run_main(["simultaneous"], process_count=3, file_count=2, tl=rec_tl, no_report=False)
    (csv_path,) = work_dir.glob("iotest_*.csv")
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["write_time", "read_time"]
    data = rows[1:]
    assert len(data) == 4
    assert [row[1] != "" for row in data] == [True, True, True, True]
    assert [row[0] != "" for row in data] == [True, True, False, False]
