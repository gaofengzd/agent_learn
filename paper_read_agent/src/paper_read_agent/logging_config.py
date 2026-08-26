"""Central logging configuration with credential redaction."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable


_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)(authorization\s*[=:]\s*bearer\s+)[^\s,;]+"),
)


def redact_text(value: object, secrets: Iterable[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text


class RedactingFilter(logging.Filter):
    """Redact configured secrets before a record reaches any handler."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.msg, self._secrets)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact_text(value, self._secrets)
                    for key, value in record.args.items()
                }
            else:
                record.args = tuple(redact_text(value, self._secrets) for value in record.args)
        return True


class RedactingFormatter(logging.Formatter):
    """Redact exception text as well as ordinary log messages."""

    def __init__(self, *args: object, secrets: Iterable[str] = (), **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._secrets = tuple(secret for secret in secrets if secret)

    def formatException(self, ei: tuple[type[BaseException], BaseException, object]) -> str:
        return redact_text(super().formatException(ei), self._secrets)


def configure_logging(level: str = "INFO", secrets: Iterable[str] = ()) -> None:
    """Configure the process root logger once with safe, concise output."""

    handler = logging.StreamHandler()
    handler.setFormatter(
        RedactingFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            secrets=secrets,
        )
    )
    handler.addFilter(RedactingFilter(secrets))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
