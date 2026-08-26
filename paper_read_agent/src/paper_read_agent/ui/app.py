"""ASGI entry point for the configured local application."""
from pathlib import Path
import os

from paper_read_agent.config import AppSettings
from paper_read_agent.ui.local_facade import LocalUIFacade
from paper_read_agent.ui.web import create_app


def _environment() -> dict[str,str]:
    values=dict(os.environ);path=Path(".env")
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:continue
            name,value=line.split("=",1)
            values.setdefault(name.strip(),value.strip().strip('"').strip("'"))
    return values


settings=AppSettings.from_env(_environment())
app=create_app(LocalUIFacade(settings))
