from home_cortex.conversation.session import parse_session, session_token, valid_session


def test_session_roundtrip_keeps_email() -> None:
    token = session_token("secret", "email", "jian@example.com")
    assert parse_session(token, "secret") == ("email", "jian@example.com")
    assert valid_session(token, "secret")
    assert parse_session(token, "other") is None
    assert parse_session("not-a-token", "secret") is None
