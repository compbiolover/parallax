"""SMTP delivery, and saying clearly when it can't happen.

Mirrors ``scoring/claude_client.py``: a missing setting degrades to "no email
sent" with a message naming the exact thing to fix, rather than a traceback or
a silent no-op. A digest that quietly fails to send is worse than one that
never existed — you stop checking the dashboard because the email is coming,
and the email isn't.
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from .render import Digest

logger = logging.getLogger(__name__)

DEFAULT_PORT = 587

NO_CONFIG = (
    "email digest not configured. Set PARALLAX_SMTP_HOST, PARALLAX_SMTP_USER, "
    "PARALLAX_SMTP_PASSWORD and PARALLAX_DIGEST_TO in the environment (see "
    ".env.example). Nothing is read from .env automatically, and cron does not "
    "inherit your shell — set them inside the crontab for scheduled runs."
)


@dataclass(frozen=True)
class MailConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str
    recipients: tuple[str, ...]
    starttls: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MailConfig | None:
        """Build from the environment, or ``None`` if it isn't fully set.

        All-or-nothing on purpose: a half-configured mailer is the case that
        fails at 6am on a machine nobody is watching.
        """
        env = os.environ if env is None else env
        host = env.get("PARALLAX_SMTP_HOST", "").strip()
        user = env.get("PARALLAX_SMTP_USER", "").strip()
        password = env.get("PARALLAX_SMTP_PASSWORD", "").strip()
        to = env.get("PARALLAX_DIGEST_TO", "").strip()
        if not (host and user and password and to):
            return None
        return cls(
            host=host,
            port=int(env.get("PARALLAX_SMTP_PORT", DEFAULT_PORT)),
            user=user,
            password=password,
            # Defaulting the From to the login address is what most providers
            # require anyway, and a mismatch is a common silent-rejection cause.
            sender=env.get("PARALLAX_DIGEST_FROM", "").strip() or user,
            recipients=tuple(r.strip() for r in to.split(",") if r.strip()),
            starttls=env.get("PARALLAX_SMTP_STARTTLS", "1") != "0",
        )


def build_message(digest: Digest, config: MailConfig) -> EmailMessage:
    """A multipart/alternative message: text first, HTML second.

    Order matters — the last part is the one a capable client shows, and the
    text part has to exist for the clients that prefer it.
    """
    message = EmailMessage()
    message["Subject"] = digest.subject
    message["From"] = config.sender
    message["To"] = ", ".join(config.recipients)
    message.set_content(digest.text)
    message.add_alternative(digest.html, subtype="html")
    return message


def send(digest: Digest, config: MailConfig | None = None, *, smtp_factory=None) -> bool:
    """Send the digest. Returns False (with a warning) rather than raising.

    ``smtp_factory`` exists so the tests can exercise the real message-building
    path without a network or a mailbox.
    """
    config = config or MailConfig.from_env()
    if config is None:
        logger.warning("%s", NO_CONFIG)
        return False

    message = build_message(digest, config)
    factory = smtp_factory or (lambda: smtplib.SMTP(config.host, config.port, timeout=30))
    try:
        with factory() as server:
            if config.starttls:
                server.starttls()
            server.login(config.user, config.password)
            server.send_message(message)
    except Exception as exc:
        logger.warning("digest send failed (%s: %s)", type(exc).__name__, exc)
        return False
    logger.info("digest sent to %s", ", ".join(config.recipients))
    return True
