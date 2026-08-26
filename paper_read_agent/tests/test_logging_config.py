import logging
import sys

from paper_read_agent.logging_config import RedactingFilter, RedactingFormatter, redact_text


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


def test_formatter_redacts_credentials_from_exception_traceback() -> None:
    secret = "traceback-secret"
    try:
        raise RuntimeError(f"request Authorization: Bearer {secret}")
    except RuntimeError:
        exc_info = sys.exc_info()
    formatter = RedactingFormatter("%(message)s", secrets=[secret])
    record = logging.LogRecord("test", logging.ERROR, __file__, 1, "failed", (), exc_info)
    rendered = formatter.format(record)
    assert secret not in rendered
    assert "Authorization: Bearer [REDACTED]" in rendered
