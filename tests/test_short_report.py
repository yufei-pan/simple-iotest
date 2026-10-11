import socket


def test_short_report_is_posted_as_json(run_main, webhook):
    url, received = webhook()
    run_main(["write"], message_end_point_address=url)
    (request,) = received()
    assert request["json"]["content"].startswith("Short Report for write mode:")
    assert request["headers"]["content-type"] == "application/json"
    # some webhook hosts (Discord) reject urllib's default Python-urllib agent
    assert request["headers"]["user-agent"].startswith("iotest/")


def test_http_error_status_is_reported(run_main, webhook, rec_tl):
    url, _ = webhook(status=403)
    run_main(["write"], message_end_point_address=url, tl=rec_tl)
    assert any("403" in m for m in rec_tl.printed)


def test_unreachable_endpoint_warns_instead_of_crashing(run_main, rec_tl):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now: connection refused
    result = run_main(["write"], message_end_point_address=f"http://127.0.0.1:{port}/", tl=rec_tl)
    assert result["outResults"]["write"]
    assert any("Failed to send short report" in m for m in rec_tl.printed)


def test_silent_endpoint_times_out_after_the_report_is_out(run_main, rec_tl, iotest_mod, monkeypatch):
    monkeypatch.setattr(iotest_mod, "SHORT_REPORT_TIMEOUT", 0.5)
    # listening but never accepting: the TCP handshake completes, no response ever comes
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    try:
        run_main(["write"], message_end_point_address=f"http://127.0.0.1:{s.getsockname()[1]}/", tl=rec_tl)
    finally:
        s.close()
    printed = rec_tl.printed
    report_at = next(i for i, m in enumerate(printed) if "Report for write mode" in m)
    warning_at = next(i for i, m in enumerate(printed) if "Failed to send short report" in m)
    assert report_at < warning_at


def test_address_without_scheme_defaults_to_http(run_main, webhook):
    # curl treated "host:port/path" as http://; existing -addr values rely on that
    url, received = webhook()
    run_main(["write"], message_end_point_address=url[len("http://"):])
    assert len(received()) == 1


def test_gives_up_after_the_first_failed_send(run_main, rec_tl):
    # rw sends two reports; an unreachable endpoint must cost one timeout, not one per report
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    run_main(["write", "read"], message_end_point_address=f"http://127.0.0.1:{port}/", tl=rec_tl)
    failures = [m for m in rec_tl.printed if "Failed to send short report" in m]
    assert len(failures) == 1
    assert "1 more" in failures[0]
