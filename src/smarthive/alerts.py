"""Notifications: console, ntfy (free phone push) and e-mail through the user's SMTP.

Nothing leaves the machine unless a channel is enabled in the config. The SMTP
password is read from the SMART_HIVE_SMTP_PASSWORD environment variable and is
never written to disk. A per-key cooldown stops alert storms.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

from .config import AlertsCfg

log = logging.getLogger(__name__)
PASSWORD_ENV = "SMART_HIVE_SMTP_PASSWORD"


@dataclass
class Alert:
    title: str
    body: str
    priority: int = 3  # ntfy scale 1-5
    tags: list[str] = field(default_factory=list)
    attachment: Path | None = None
    key: str = ""  # cooldown key, e.g. "hornet:hive1"


def _ascii_header(text: str) -> str:
    # HTTP headers must be latin-1; ntfy also accepts RFC 2047 but keep it simple.
    return text.encode("latin-1", "replace").decode("latin-1")


class Notifier:
    def __init__(self, cfg: AlertsCfg, dry_run: bool = False, clock=time.monotonic):
        self.cfg = cfg
        self.dry_run = dry_run
        self.clock = clock
        self._last: dict[str, float] = {}
        self.sent: list[tuple[str, Alert]] = []  # (channel, alert), for tests and logs

    def channels(self) -> list[str]:
        out = ["console"]
        if self.cfg.ntfy.enabled:
            out.append("ntfy")
        if self.cfg.email.enabled:
            out.append("email")
        return out

    def send(self, alert: Alert, cooldown_s: float = 0.0) -> bool:
        """Send on every enabled channel. Returns False if suppressed by cooldown."""
        now = self.clock()
        if alert.key and cooldown_s > 0:
            last = self._last.get(alert.key)
            if last is not None and now - last < cooldown_s:
                log.info("alert %s suppressed (cooldown)", alert.key)
                return False
        if alert.key:
            self._last[alert.key] = now
        for channel in self.channels():
            try:
                getattr(self, f"_send_{channel}")(alert)
                self.sent.append((channel, alert))
            except Exception as exc:  # a broken channel must not stop monitoring
                log.error("alert via %s failed: %s", channel, exc)
        return True

    # -- channels ------------------------------------------------------------
    def _send_console(self, alert: Alert) -> None:
        extra = f" [{alert.attachment}]" if alert.attachment else ""
        print(f"[ALERT] {alert.title}: {alert.body}{extra}", flush=True)

    def _send_ntfy(self, alert: Alert) -> None:
        url = f"{self.cfg.ntfy.server.rstrip('/')}/{self.cfg.ntfy.topic}"
        headers = {
            "Title": _ascii_header(alert.title),
            "Priority": str(alert.priority),
            "User-Agent": "smart-hive/0.1",
        }
        if alert.tags:
            headers["Tags"] = ",".join(alert.tags)
        if alert.attachment and Path(alert.attachment).exists():
            data = Path(alert.attachment).read_bytes()
            headers["Filename"] = Path(alert.attachment).name
            headers["Message"] = _ascii_header(alert.body)
            method = "PUT"
        else:
            data = alert.body.encode("utf-8")
            method = "POST"
        if self.dry_run:
            print(f"[dry-run] ntfy {method} {url} {headers}")
            return
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()

    def _send_email(self, alert: Alert) -> None:
        cfg = self.cfg.email
        msg = EmailMessage()
        msg["Subject"] = alert.title
        msg["From"] = cfg.sender or cfg.username
        msg["To"] = ", ".join(cfg.to)
        msg.set_content(alert.body)
        if alert.attachment and Path(alert.attachment).exists():
            msg.add_attachment(
                Path(alert.attachment).read_bytes(),
                maintype="image",
                subtype="jpeg",
                filename=Path(alert.attachment).name,
            )
        if self.dry_run:
            print(f"[dry-run] e-mail to {cfg.to}: {alert.title}")
            return
        password = os.environ.get(PASSWORD_ENV, "")
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as smtp:
            if cfg.starttls:
                smtp.starttls(context=ssl.create_default_context())
            if cfg.username:
                if not password:
                    raise RuntimeError(f"set {PASSWORD_ENV} to send e-mail")
                smtp.login(cfg.username, password)
            smtp.send_message(msg)
