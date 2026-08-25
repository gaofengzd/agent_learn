import logging

from paper_read_agent.logging_config import RedactingFilter, redact_text


def test_redacts_known_secret() -> None:
    assert redact_text("token=very-secret", ["very-secret"]) == "token=[REDACTED]"


def test_redacts_common_credential_patterns() -> None:
    assert "actual-key" not in redact_text("api_key=actual-key")
    assert "bearer-token" not in redact_text("Authorization: Bearer bearer-token")


def test_filter_redacts_message_and_arguments() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="key=%s api_key=inline-secret",
        args=("configured-secret",),
        exc_info=None,
    )

    allowed = RedactingFilter(["configured-secret"]).filter(record)
    rendered = record.getMessage()

    assert allowed is True
    assert "configured-secret" not in rendered
    assert "inline-secret" not in rendered
    assert rendered.count("[REDACTED]") == 2
