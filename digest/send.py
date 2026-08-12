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
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage

from .render import Digest

logger = logging.getLogger(__name__)

DEFAULT_PORT = 587
IMPLICIT_TLS_PORT = 465  # TLS from the first byte; no STARTTLS handshake
TIMEOUT_S = 30

# Sending a password in the clear is defensible to a relay on this machine and
# nowhere else. `PARALLAX_SMTP_STARTTLS=0` is documented for exactly that case.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

NO_CONFIG = (
    "email digest not configured. Set PARALLAX_SMTP_HOST, PARALLAX_SMTP_USER, "
    "PARALLAX_SMTP_PASSWORD and PARALLAX_DIGEST_TO in the environment (see "
    ".env.example). Nothing is read from .env automatically, and cron does not "
    "inherit your shell — set them inside the crontab for scheduled runs."
)

PLAINTEXT_REMOTE = (
    "refusing to send: PARALLAX_SMTP_STARTTLS=0 disables TLS, and {host} is not "
    "on this machine, so the SMTP password would cross the network in the clear. "
    "That option exists for a local relay only. Remove it, or point the digest at "
    "a relay on localhost."
)

CERT_FAILED = (
    "digest send failed: could not verify the TLS certificate for {host} ({exc}). "
    "This is what a misconfigured server looks like — and also what an intercepted "
    "connection looks like. The SMTP password was NOT sent. Do not work around it "
    "by disabling verification; fix the server's certificate, or use a host whose "
    "certificate validates."
)


def _port(env) -> int:
    """The SMTP port, falling back rather than raising.

    ``.env.example`` is a file of ``KEY=`` lines, so blanking the port is the
    natural thing to do — and every other variable treats empty as unset.
    ``int("")`` here used to be an uncaught traceback out of the CLI and a
    ``ValueError`` in the daily report, which is exactly what this module's
    docstring promises not to do.
    """
    raw = env.get("PARALLAX_SMTP_PORT", "").strip()
    if raw.isdigit():
        return int(raw)
    if raw:
        logger.warning("PARALLAX_SMTP_PORT=%r is not a port number; using %d", raw, DEFAULT_PORT)
    return DEFAULT_PORT


@dataclass(frozen=True)
class MailConfig:
    host: str
    port: int
    user: str
    # Kept out of the auto-generated repr. Nothing logs this object today, but a
    # dataclass repr is one `logger.debug(config)` away from putting a live mail
    # password in a log file — and for a Gmail app password that is full mailbox
    # access. Cheap to prevent, tedious to clean up after.
    password: str = field(repr=False)
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
            port=_port(env),
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


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in LOOPBACK_HOSTS


def _connect(config: MailConfig):
    """Open the connection, with certificate verification where TLS applies.

    Port 465 is implicit TLS — encrypted from the first byte, so it is wrapped
    at connect time and never sees a STARTTLS handshake. Everything else
    connects in the clear and upgrades in ``send``.
    """
    if config.port == IMPLICIT_TLS_PORT:
        return smtplib.SMTP_SSL(
            config.host, config.port, timeout=TIMEOUT_S, context=ssl.create_default_context()
        )
    return smtplib.SMTP(config.host, config.port, timeout=TIMEOUT_S)


def send(digest: Digest, config: MailConfig | None = None, *, smtp_factory=None) -> str | None:
    """Send the digest. Returns ``None`` on success, or the reason it failed.

    The reason is returned rather than a bare ``False`` because the caller puts
    it in the daily report, and there are four quite different ways this fails:
    unconfigured, a refused cleartext send, a certificate that would not verify,
    and everything else. Reporting "not configured" for a refused connection
    sends you to edit environment variables that were already correct.

    The password crosses this connection, so the TLS has to be *authenticated*
    TLS. ``smtplib``'s ``starttls()`` defaults to ``ssl._create_stdlib_context()``
    when given no context, which is ``check_hostname=False`` and
    ``verify_mode=CERT_NONE`` — encryption with no idea who is on the other end,
    and no error to tell you so. ``ssl.create_default_context()`` is passed
    explicitly for that reason. This is true on every Python in range (3.11
    through 3.14 all still take that default), so it is not a version to grow
    out of.

    ``smtp_factory`` exists so the tests can exercise the real message-building
    path without a network or a mailbox.
    """
    config = config or MailConfig.from_env()
    if config is None:
        logger.warning("%s", NO_CONFIG)
        return NO_CONFIG

    # Refused before the connection opens, not warned about after the password
    # is already gone. Encryption off is a deliberate setting; sending a
    # credential across a network in the clear is a different thing entirely.
    if not config.starttls and config.port != IMPLICIT_TLS_PORT and not _is_loopback(config.host):
        reason = PLAINTEXT_REMOTE.format(host=config.host)
        logger.warning("%s", reason)
        return reason

    message = build_message(digest, config)
    factory = smtp_factory or (lambda: _connect(config))
    try:
        with factory() as server:
            if config.starttls and config.port != IMPLICIT_TLS_PORT:
                server.starttls(context=ssl.create_default_context())
            server.login(config.user, config.password)
            server.send_message(message)
    except ssl.SSLCertVerificationError as exc:
        # Called out separately because the generic message below reads like a
        # server problem, and this one may not be.
        reason = CERT_FAILED.format(host=config.host, exc=exc)
        logger.warning("%s", reason)
        return reason
    except Exception as exc:
        reason = f"digest send failed ({type(exc).__name__}: {exc})"
        logger.warning("%s", reason)
        return reason
    logger.info("digest sent to %s", ", ".join(config.recipients))
    return None
