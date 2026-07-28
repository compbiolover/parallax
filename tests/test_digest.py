"""The daily brief: honest arithmetic, symmetric framing, and email that works.

The digest is the only part of Parallax the author will actually read most
days, so the failure that matters is not a crash — it is a number that reads as
more certain, or a framing that reads as more one-sided, than the data supports.
"""

from __future__ import annotations

import logging
import re
import ssl

import pytest

from daily import runner
from daily.runner import DailyConfig
from digest.render import _delta, _label, _own_first, build_digest, render_html, render_text
from digest.send import MailConfig, build_message, send

FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "sanctity"]


def _payload(**over) -> dict:
    base = {
        "generated_utc": "2026-07-28T11:02:00+00:00",
        "foundations": FOUNDATIONS,
        "diets": [
            {"id": "self", "doc_count": 128,
             "profile": {"care": .31, "fairness": .27, "loyalty": .14,
                         "authority": .16, "sanctity": .12},
             "summary": "Harm and procedural fairness."},
            {"id": "modeled_ce", "doc_count": 96,
             "profile": {"care": .19, "fairness": .18, "loyalty": .24,
                         "authority": .21, "sanctity": .18},
             "summary": "Institutional trust and obligation."},
        ],
        "comparison": {"pair": ["self", "modeled_ce"], "jsd": 0.0841,
                       "log_ratios": {"care": .49, "loyalty": -.54}},
        "history": [
            {"date": "2026-07-27", "jsd_window": 0.078, "window_days": 7},
            {"date": "2026-07-28", "jsd_window": 0.0841, "window_days": 7},
        ],
        "history_window_days": 7,
        "blindspots": [],
        "executive_summary": "Same story, different registers.",
        "summary_method": "claude-sonnet-5",
        "caveat": "Read every number as a noisy estimate, never ground truth.",
    }
    return {**base, **over}


def _spot(dominant, other, label="a cluster"):
    return {"label": label, "dominant_diet": dominant, "other_diet": other,
            "size": 12, "dominant_share": 0.9, "representative_titles": ["A headline"]}


# -- unscored is not zero ---------------------------------------------------


def test_no_history_says_so_rather_than_showing_no_movement():
    delta, reason = _delta([])
    assert delta is None
    assert "no divergence recorded yet" in reason


def test_a_first_point_has_no_delta_rather_than_a_zero_one():
    """`+0.000` would read as 'nothing moved'. Nothing moved is a claim; having
    no previous day is the absence of one."""
    delta, reason = _delta([{"jsd_window": 0.08}])
    assert delta is None
    assert "first recorded point" in reason


def test_delta_is_newest_minus_previous():
    delta, _ = _delta([{"jsd_window": 0.070}, {"jsd_window": 0.078}])
    assert delta == pytest.approx(0.008)


def test_points_without_a_windowed_value_are_skipped_not_counted_as_zero():
    history = [{"jsd_window": 0.07}, {"jsd_window": None}, {"jsd_window": 0.09}]
    delta, _ = _delta(history)
    assert delta == pytest.approx(0.02)


def test_a_single_usable_point_among_nulls_still_refuses_a_delta():
    assert _delta([{"jsd_window": None}, {"jsd_window": 0.08}])[0] is None


# -- the subject line has to be true on its own -----------------------------


def test_subject_carries_the_headline_and_its_movement():
    """On a phone the subject is often the whole reading, so it must not imply
    more than the body says."""
    subject = build_digest(_payload()).subject
    assert "0.084" in subject
    assert "+0.006" in subject
    assert "2026-07-28" in subject


def test_subject_omits_movement_when_there_is_none_to_report():
    subject = build_digest(_payload(history=[])).subject
    assert "0.084" in subject
    assert "+" not in subject.split("divergence")[-1]


def test_subject_says_so_when_there_is_nothing_to_compare():
    assert "not enough data" in build_digest(_payload(comparison=None)).subject


# -- symmetry ---------------------------------------------------------------


def test_the_authors_own_blindspots_come_first():
    """Equal styling is not equal prominence on a phone: whatever is second gets
    scrolled past, and the author's own blindspots are the direction the whole
    confirmation-bias guard exists to protect."""
    spots = [_spot("self", "modeled_ce", "theirs"), _spot("modeled_ce", "self", "mine")]
    assert [s["label"] for s in _own_first(spots, "self")] == ["mine", "theirs"]


def test_both_directions_are_rendered():
    spots = [_spot("self", "modeled_ce", "they missed this"),
             _spot("modeled_ce", "self", "I missed this")]
    html = render_html(_payload(blindspots=spots), own_diet="self")
    assert "they missed this" in html
    assert "I missed this" in html


def test_ordering_is_stable_when_no_diet_is_named_as_the_authors():
    spots = [_spot("self", "modeled_ce", "one"), _spot("modeled_ce", "self", "two")]
    assert _own_first(spots, None) == spots


def test_neither_diet_is_described_in_pejorative_terms():
    """§0: binding foundations are sincere moral commitments, not deficits. The
    rendering must not editorialise where the data doesn't."""
    html = render_html(_payload(blindspots=[_spot("modeled_ce", "self")]), own_diet="self")
    assert not re.search(r"\b(bias(ed)?|extreme|fringe|echo chamber|failure)\b", html, re.I)


def test_the_blindspot_note_declines_to_rank_the_stories():
    html = render_html(_payload(blindspots=[_spot("modeled_ce", "self")]), own_diet="self")
    assert "not a judgement about which story mattered more" in html


# -- uncertainty travels with the numbers -----------------------------------


def test_the_caveat_is_always_present():
    """The temptation on a small screen is to trim it. It is the one line that
    keeps a noisy estimate from reading as a measurement."""
    assert "noisy estimate" in render_html(_payload())
    assert "noisy estimate" in render_text(_payload())


def test_liberty_carries_its_own_weaker_provenance():
    payload = _payload(liberty={
        "scorer": "claude-liberty/claude-sonnet-5",
        "diets": {"self": {"mean": .31, "salient_share": .18, "docs_scored": 88,
                           "docs_total": 128, "coverage": .69, "thin": False}},
        "gap": 0.1,
    })
    html = render_html(payload)
    assert "least corroborated" in html
    assert "69% of docs scored" in html      # coverage, not just the mean


def test_a_fairness_split_with_no_evidence_is_not_shown_as_a_ratio():
    payload = _payload(fairness_split={
        "diets": {"self": {"equality": None, "proportionality": None,
                           "docs_split": 0, "docs_total": 128, "coverage": 0.0,
                           "thin": True, "leans": None}},
        "gap": None,
    })
    assert "not enough split-terms" in render_html(payload)


# -- email mechanics --------------------------------------------------------


def test_the_html_loads_nothing_from_the_network():
    """A remote image is a tracking pixel to a mail client, and a blocked one is
    a broken chart. Everything has to be markup."""
    html = render_html(_payload(blindspots=[_spot("modeled_ce", "self")]))
    assert "http://" not in html
    assert "https://" not in html
    assert "<img" not in html.lower()


def test_the_html_contains_no_script():
    html = render_html(_payload())
    assert "<script" not in html.lower()
    assert "javascript:" not in html.lower()


def test_untrusted_text_is_escaped():
    """Cluster labels and summaries come from feeds and from a model. Neither is
    markup."""
    payload = _payload(
        executive_summary="<script>alert(1)</script> & then",
        blindspots=[_spot("modeled_ce", "self", '"><b>x</b>')],
    )
    html = render_html(payload, own_diet="self")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "<b>x</b>" not in html


def test_an_empty_payload_renders_instead_of_crashing():
    """The first morning after a failed ingest still sends an email; it should
    say there is nothing to compare, not raise inside cron."""
    digest = build_digest({"generated_utc": "2026-07-28T00:00:00+00:00"})
    assert "not enough" in digest.subject.lower()
    assert "Parallax" in digest.html
    assert digest.text.strip()


def test_the_preheader_summarises_without_the_body():
    preheader = build_digest(_payload(blindspots=[_spot("modeled_ce", "self")])).preheader
    assert "0.084" in preheader
    assert "1 blindspot" in preheader


def test_diet_ids_render_initialisms_in_caps():
    assert _label("modeled_ce") == "Modeled CE"
    assert _label("self") == "Self"


# -- sending ----------------------------------------------------------------


ENV = {
    "PARALLAX_SMTP_HOST": "smtp.example.com",
    "PARALLAX_SMTP_USER": "me@example.com",
    "PARALLAX_SMTP_PASSWORD": "secret",
    "PARALLAX_DIGEST_TO": "me@example.com",
}


def test_config_needs_every_setting():
    """All-or-nothing on purpose: a half-configured mailer is the case that
    fails at 6am on a machine nobody is watching."""
    for missing in ENV:
        assert MailConfig.from_env({k: v for k, v in ENV.items() if k != missing}) is None
    assert MailConfig.from_env(ENV) is not None


def test_sender_defaults_to_the_login_address():
    """Most providers reject a From that doesn't match the authenticated user,
    and they do it silently."""
    assert MailConfig.from_env(ENV).sender == "me@example.com"


def test_recipients_split_on_commas():
    config = MailConfig.from_env({**ENV, "PARALLAX_DIGEST_TO": "a@x.com, b@x.com"})
    assert config.recipients == ("a@x.com", "b@x.com")


def test_message_carries_both_parts_with_html_last():
    """multipart/alternative shows the last part a client can render; the text
    part still has to exist for the ones that prefer it."""
    message = build_message(build_digest(_payload()), MailConfig.from_env(ENV))
    subtypes = [p.get_content_subtype() for p in message.iter_parts()]
    assert subtypes == ["plain", "html"]
    assert "divergence" in message["Subject"]


def test_send_without_configuration_warns_and_returns_false(monkeypatch, caplog):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    with caplog.at_level(logging.WARNING):
        assert send(build_digest(_payload())) is False
    assert "not configured" in caplog.text
    assert "crontab" in caplog.text          # cron doesn't inherit your shell


class _SMTP:
    """Records the ordered calls the sender makes, including the TLS context."""

    def __init__(self, calls):
        self.calls = calls

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def starttls(self, context=None):
        self.calls.append(f"starttls:{type(context).__name__}")
        if context is not None:
            self.calls.append(f"verify:{context.verify_mode.name}")
            self.calls.append(f"hostname:{context.check_hostname}")

    def login(self, user, password): self.calls.append(f"login:{user}")
    def send_message(self, message): self.calls.append(f"send:{message['To']}")


def test_send_uses_starttls_login_and_send():
    calls = []
    config = MailConfig.from_env(ENV)
    assert send(build_digest(_payload()), config,
                smtp_factory=lambda: _SMTP(calls)) is True
    assert calls[0] == "starttls:SSLContext"
    assert calls[-2:] == ["login:me@example.com", "send:me@example.com"]


# -- the credential crosses this connection ---------------------------------


def test_starttls_gets_a_verifying_context():
    """The finding this exists to prevent: smtplib's starttls() with no context
    uses ssl._create_stdlib_context(), which is CERT_NONE + check_hostname
    False. That is encryption with no idea who is on the other end, and no
    error to say so — an on-path attacker presents any certificate and reads
    the AUTH exchange that follows in the clear."""
    calls = []
    send(build_digest(_payload()), MailConfig.from_env(ENV),
         smtp_factory=lambda: _SMTP(calls))
    assert "verify:CERT_REQUIRED" in calls
    assert "hostname:True" in calls
    # and the password only moves after the handshake is verified
    assert calls.index("starttls:SSLContext") < calls.index("login:me@example.com")


def test_the_stdlib_default_really_is_unverified():
    """Pins the premise. If a future Python makes starttls() safe by default the
    explicit context becomes redundant rather than load-bearing — but until this
    assertion fails, it is load-bearing."""
    assert ssl._create_stdlib_context().verify_mode is ssl.CERT_NONE
    assert ssl._create_stdlib_context().check_hostname is False
    assert ssl.create_default_context().verify_mode is ssl.CERT_REQUIRED


def test_plaintext_to_a_remote_host_is_refused_before_connecting(caplog):
    """Disabling encryption is a deliberate setting. Sending a password across
    a network in the clear is a different thing, and it is refused rather than
    warned about after the fact."""
    opened = []
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_STARTTLS": "0"})
    with caplog.at_level(logging.WARNING):
        result = send(build_digest(_payload()), config,
                      smtp_factory=lambda: opened.append(1) or _SMTP([]))
    assert result is False
    assert opened == []                       # never dialled
    assert "would cross the network in the clear" in caplog.text


def test_plaintext_to_a_local_relay_is_allowed():
    """The documented use for STARTTLS=0 — a relay on this machine, where there
    is no network for anyone to sit on."""
    calls = []
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_STARTTLS": "0",
                                  "PARALLAX_SMTP_HOST": "localhost"})
    assert send(build_digest(_payload()), config,
                smtp_factory=lambda: _SMTP(calls)) is True
    assert not any(c.startswith("starttls") for c in calls)
    assert "login:me@example.com" in calls


def test_implicit_tls_port_skips_the_starttls_upgrade():
    """465 is TLS from the first byte; a STARTTLS handshake on top of it is an
    error, not a belt-and-braces."""
    calls = []
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_PORT": "465"})
    assert send(build_digest(_payload()), config,
                smtp_factory=lambda: _SMTP(calls)) is True
    assert not any(c.startswith("starttls") for c in calls)


def test_implicit_tls_is_not_refused_when_starttls_is_off():
    """465 with STARTTLS=0 is a correct configuration, not a cleartext one."""
    calls = []
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_PORT": "465",
                                  "PARALLAX_SMTP_STARTTLS": "0"})
    assert send(build_digest(_payload()), config,
                smtp_factory=lambda: _SMTP(calls)) is True


def test_a_certificate_failure_is_reported_as_its_own_thing(caplog):
    """A verification failure is what a misconfigured server looks like and
    also what an interception looks like. Folding it into the generic 'send
    failed' invites disabling verification to make the message go away."""
    def _boom():
        raise ssl.SSLCertVerificationError("hostname mismatch")

    with caplog.at_level(logging.WARNING):
        assert send(build_digest(_payload()), MailConfig.from_env(ENV),
                    smtp_factory=_boom) is False
    assert "could not verify the TLS certificate" in caplog.text
    assert "password was NOT sent" in caplog.text
    assert "Do not work around it" in caplog.text


def test_a_send_failure_is_reported_not_raised(monkeypatch, caplog):
    """The daily run must survive a mail outage; the step reports it."""
    def _boom():
        raise OSError("connection refused")

    with caplog.at_level(logging.WARNING):
        assert send(build_digest(_payload()), MailConfig.from_env(ENV),
                    smtp_factory=_boom) is False
    assert "connection refused" in caplog.text


# -- wiring into the daily run ----------------------------------------------


def test_digest_is_off_unless_switched_on():
    """Every other step defaults on. This one needs credentials nobody has on a
    first run, and a step that fails every morning teaches you to ignore the
    report that is supposed to tell you when something broke."""
    assert "digest" not in DailyConfig.from_settings({}).steps
    assert "digest" in DailyConfig.from_settings({"digest": {"enabled": True}}).steps


def test_own_diet_comes_from_settings():
    cfg = DailyConfig.from_settings({"digest": {"enabled": True, "own_diet": "self"}})
    assert cfg.own_diet == "self"


def test_an_unsendable_digest_fails_the_step_rather_than_passing_quietly(monkeypatch):
    """Switched on and not sending is exactly what you need to be told about —
    you stop checking the dashboard because the email is coming, and it isn't."""
    monkeypatch.setattr("digest.send.send", lambda *a, **k: False)
    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    with pytest.raises(RuntimeError, match="not configured"):
        runner._step_digest(object(), DailyConfig(own_diet="self"))


def test_a_sent_digest_reports_the_subject(monkeypatch):
    monkeypatch.setattr("digest.send.send", lambda *a, **k: True)
    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    detail = runner._step_digest(object(), DailyConfig(own_diet="self"))
    assert "0.084" in detail
