"""The daily brief: honest arithmetic, symmetric framing, and email that works.

The digest is the only part of Parallax the author will actually read most
days, so the failure that matters is not a crash — it is a number that reads as
more certain, or a framing that reads as more one-sided, than the data supports.
"""

from __future__ import annotations

import logging
import pathlib
import re
import ssl

import pytest

from daily import runner
from daily.runner import DailyConfig
from digest.render import _delta, _label, _own_first, build_digest, render_html, render_text
from digest.send import NO_CONFIG, MailConfig, build_message, send

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
            {"date": "2026-07-27", "jsd_cumulative": 0.078, "jsd_window": 0.061,
             "window_days": 7},
            {"date": "2026-07-28", "jsd_cumulative": 0.0841, "jsd_window": 0.095,
             "window_days": 7},
        ],
        "history_window_days": 7,
        "blindspots": [],
        "executive_summary": "Same story, different registers.",
        "summary_method": "claude-sonnet-5",
        "caveat": "Read every number as a noisy estimate, never ground truth.",
    }
    return {**base, **over}


def _spot(dominant, other, label="a cluster", titles=("A headline about something",),
          cluster_id=0, size=12):
    """One blindspot cluster as the exporter serializes it.

    ``label`` is the c-TF-IDF readout the email no longer prints; it stays in
    the fixture because the payload still carries it and because a test that it
    is *not* printed needs it to be there.
    """
    return {"cluster_id": cluster_id, "label": label, "dominant_diet": dominant,
            "other_diet": other, "size": size, "dominant_share": 0.9,
            "counts": {dominant: size - 1, other: 1},
            "representative_titles": list(titles)}


# -- unscored is not zero ---------------------------------------------------


def test_no_history_says_so_rather_than_showing_no_movement():
    delta, reason, _ = _delta([])
    assert delta is None
    assert "no divergence recorded yet" in reason


def test_a_first_point_has_no_delta_rather_than_a_zero_one():
    """`+0.000` would read as 'nothing moved'. Nothing moved is a claim; having
    no previous day is the absence of one."""
    delta, reason, _ = _delta([{"jsd_cumulative": 0.08}])
    assert delta is None
    assert "first recorded point" in reason


def test_delta_is_newest_minus_previous():
    delta, _, _ = _delta([{"jsd_cumulative": 0.070}, {"jsd_cumulative": 0.078}])
    assert delta == pytest.approx(0.008)


def test_delta_reads_the_same_series_as_the_headline():
    """The headline is comparison.jsd, computed over the whole corpus, so the
    movement under it has to be the cumulative series. Reading jsd_window put a
    delta from a different basis under the number — a figure that could even
    move the opposite way, since the cumulative basis is damped by design."""
    history = [{"date": "2026-07-27", "jsd_cumulative": 0.070, "jsd_window": 0.200},
               {"date": "2026-07-28", "jsd_cumulative": 0.078, "jsd_window": 0.100}]
    delta, _, since = _delta(history)
    assert delta == pytest.approx(0.008)      # not -0.100
    assert since == "2026-07-27"


def test_the_delta_names_the_date_it_compares_against():
    """Snapshots are one row per UTC date but nothing makes them consecutive —
    a machine that was off leaves a gap. 'since yesterday' was a claim the data
    could not support."""
    html = render_html(_payload())
    assert "since 2026-07-27" in html
    assert "yesterday" not in html


def test_points_without_a_windowed_value_are_skipped_not_counted_as_zero():
    history = [{"jsd_cumulative": 0.07}, {"jsd_cumulative": None},
               {"jsd_cumulative": 0.09}]
    delta, _, _ = _delta(history)
    assert delta == pytest.approx(0.02)


def test_a_single_usable_point_among_nulls_still_refuses_a_delta():
    assert _delta([{"jsd_cumulative": None}, {"jsd_cumulative": 0.08}])[0] is None


# -- the subject line has to be true on its own -----------------------------


def test_subject_carries_the_headline_and_its_movement():
    """On a phone the subject is often the whole reading, so it must not imply
    more than the body says."""
    subject = build_digest(_payload()).subject
    assert "0.084" in subject
    assert "+0.006" in subject
    assert "2026-07-28" in subject


def test_subject_omits_movement_when_there_is_none_to_report():
    """Asserted exactly: `"+" not in ...` would still pass if a negative delta
    leaked through."""
    assert build_digest(_payload(history=[])).subject == (
        "Parallax 2026-07-28 — divergence 0.084")


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
    spots = [_spot("self", "modeled_ce", titles=["A story they missed"]),
             _spot("modeled_ce", "self", titles=["A story I missed"])]
    html = render_html(_payload(blindspots=spots), own_diet="self")
    assert "A story they missed" in html
    assert "A story I missed" in html
    assert "Self covered" in html and "Modeled CE covered" in html


def test_the_cluster_label_is_not_what_the_reader_is_shown():
    """c-TF-IDF labels are a technical readout — "kidney stone · bret ·
    institutional" over three unrelated headlines. The card is titled by theme."""
    payload = _payload(blindspots=[
        _spot("modeled_ce", "self", "kidney stone · bret · institutional",
              titles=["Bret Michaels shares a health update after a procedure"])])
    html = render_html(payload, own_diet="self")
    assert "kidney stone · bret" not in html
    assert "Health &amp; medicine" in html


def test_each_direction_keeps_its_own_room_on_the_page():
    """Caps are per direction, not per section. A day when one diet clusters
    noisily must not push the other diet's blindspots off the email — that is
    the half the confirmation-bias guard exists to protect."""
    subjects = ["church congregation", "election ballot voters", "hurricane evacuation",
                "cancer treatment patients", "tariffs and wages"]
    spots = [_spot("modeled_ce", "self", cluster_id=i, size=20 - i,
                   titles=[f"A story about {words} and several more words"])
             for i, words in enumerate(subjects)]
    spots += [_spot("self", "modeled_ce", cluster_id=99, size=2,
                    titles=["A story about carbon emissions and ocean warming"])]
    html = render_html(_payload(blindspots=spots), own_diet="self")
    # five themes on one side, the smallest theme on the other — the author's
    # own blindspot still gets its heading and its card
    assert "Self covered · Modeled CE barely did" in html
    assert "Climate, energy &amp; environment" in html
    assert "2 more themes here" in html


def test_themes_the_email_leaves_out_are_named_not_merely_counted():
    """A theme the reader is not shown is otherwise indistinguishable from one
    that was never found — the failure the whole idea exists to avoid."""
    subjects = [("church congregation", "Faith &amp; the church"),
                ("election ballot voters", "Elections"),
                ("hurricane evacuation", "Disasters"),
                ("cancer treatment patients", "Health &amp; medicine")]
    spots = [_spot("modeled_ce", "self", cluster_id=i, size=10 - i,
                   titles=[f"A story about {words} and more words here"])
             for i, (words, _) in enumerate(subjects)]
    html = render_html(_payload(blindspots=spots), own_diet="self")
    assert "1 more theme here: Health &amp; medicine" in html
    text = render_text(_payload(blindspots=spots), own_diet="self")
    assert "+1 more theme here: Health & medicine" in text


def test_themes_already_in_the_payload_are_the_ones_rendered():
    """The cluster run names themes once and persists them. The email reads that
    naming rather than redoing it, or the two surfaces can disagree on a day the
    model was reachable for one of them."""
    payload = _payload(
        blindspots=[_spot("modeled_ce", "self", titles=["A church story"])],
        blindspot_themes=[{
            "key": "faith", "title": "Life of the church", "dominant_diet": "modeled_ce",
            "other_diet": "self", "cluster_count": 2, "story_count": 7,
            "one_sided": 0.93, "stories": ["A church story"], "method": "claude",
        }],
    )
    html = render_html(payload, own_diet="self")
    assert "Life of the church" in html
    assert "7 stories · 2 clusters · 93% one-sided" in html


def test_ordering_is_stable_when_no_diet_is_named_as_the_authors():
    spots = [_spot("self", "modeled_ce", "one"), _spot("modeled_ce", "self", "two")]
    assert _own_first(spots, None) == spots


def test_neither_diet_is_described_in_pejorative_terms():
    """§0: binding foundations are sincere moral commitments, not deficits.

    Scoped to the module's own copy rather than the rendered output: cluster
    labels come from real headlines, so grepping the render would assert a
    property of the news rather than of the template."""
    import digest.render as mod

    copy = "\n".join(re.findall(r'"[^"]{12,}"', pathlib.Path(mod.__file__).read_text()))
    assert not re.search(r"\b(bias(ed)?|extreme|fringe|echo chamber)\b", copy, re.I)


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
    a broken chart. Asserted on the tags that *fetch* — a bare URL inside a
    headline is a property of the data, not a load."""
    html = render_html(_payload(blindspots=[_spot("modeled_ce", "self")])).lower()
    for fetching in ("<img", "<link", "<iframe", "url(", "background-image", "@import"):
        assert fetching not in html


def test_the_html_contains_no_script():
    html = render_html(_payload())
    assert "<script" not in html.lower()
    assert "javascript:" not in html.lower()


def test_untrusted_text_is_escaped():
    """Cluster labels and summaries come from feeds and from a model. Neither is
    markup."""
    payload = _payload(
        executive_summary="<script>alert(1)</script> & then",
        blindspots=[_spot("modeled_ce", "self", titles=['"><b>x</b>'])],
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
        assert send(build_digest(_payload())) is not None
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
                smtp_factory=lambda: _SMTP(calls)) is None
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
    assert result is not None
    assert opened == []                       # never dialled
    assert "would cross the network in the clear" in caplog.text


def test_plaintext_to_a_local_relay_is_allowed():
    """The documented use for STARTTLS=0 — a relay on this machine, where there
    is no network for anyone to sit on."""
    calls = []
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_STARTTLS": "0",
                                  "PARALLAX_SMTP_HOST": "localhost"})
    assert send(build_digest(_payload()), config,
                smtp_factory=lambda: _SMTP(calls)) is None
    assert not any(c.startswith("starttls") for c in calls)
    assert "login:me@example.com" in calls


def test_implicit_tls_port_skips_the_starttls_upgrade():
    """465 is TLS from the first byte; a STARTTLS handshake on top of it is an
    error, not a belt-and-braces."""
    calls = []
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_PORT": "465"})
    assert send(build_digest(_payload()), config,
                smtp_factory=lambda: _SMTP(calls)) is None
    assert not any(c.startswith("starttls") for c in calls)


def test_implicit_tls_is_not_refused_when_starttls_is_off():
    """465 with STARTTLS=0 is a correct configuration, not a cleartext one."""
    calls = []
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_PORT": "465",
                                  "PARALLAX_SMTP_STARTTLS": "0"})
    assert send(build_digest(_payload()), config,
                smtp_factory=lambda: _SMTP(calls)) is None


def test_a_certificate_failure_is_reported_as_its_own_thing(caplog):
    """A verification failure is what a misconfigured server looks like and
    also what an interception looks like. Folding it into the generic 'send
    failed' invites disabling verification to make the message go away."""
    def _boom():
        raise ssl.SSLCertVerificationError("hostname mismatch")

    with caplog.at_level(logging.WARNING):
        assert send(build_digest(_payload()), MailConfig.from_env(ENV),
                    smtp_factory=_boom) is not None
    assert "could not verify the TLS certificate" in caplog.text
    assert "password was NOT sent" in caplog.text
    assert "Do not work around it" in caplog.text


def test_a_send_failure_is_reported_not_raised(monkeypatch, caplog):
    """The daily run must survive a mail outage; the step reports it."""
    def _boom():
        raise OSError("connection refused")

    with caplog.at_level(logging.WARNING):
        assert send(build_digest(_payload()), MailConfig.from_env(ENV),
                    smtp_factory=_boom) is not None
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
    monkeypatch.setattr("digest.send.send", lambda *a, **k: NO_CONFIG)
    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    with pytest.raises(RuntimeError, match="not configured"):
        runner._step_digest(object(), DailyConfig(own_diet="self"))


def test_the_step_reports_the_actual_cause_not_a_generic_one(monkeypatch):
    """send() fails for four different reasons and the daily report is what
    gets read. Saying "not configured" when the connection was refused sends
    you to edit environment variables that were already correct."""
    monkeypatch.setattr("digest.send.send",
                        lambda *a, **k: "digest send failed (OSError: connection refused)")
    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    with pytest.raises(RuntimeError, match="connection refused"):
        runner._step_digest(object(), DailyConfig(own_diet="self"))


def test_a_sent_digest_reports_the_subject(monkeypatch):
    monkeypatch.setattr("digest.send.send", lambda *a, **k: None)
    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    detail = runner._step_digest(object(), DailyConfig(own_diet="self"))
    assert "0.084" in detail


# -- panels the first round left untested -----------------------------------


def _history(n, start=0.05, step=0.002):
    return [{"date": f"2026-07-{d:02d}", "jsd_cumulative": start + i * step,
             "jsd_window": start + i * step, "window_days": 7}
            for i, d in enumerate(range(1, n + 1))]


def test_the_sparkline_shows_only_the_most_recent_window():
    """A year of snapshots at phone width is a grey smear. SPARK_DAYS columns
    is what fits."""
    from digest.render import SPARK_DAYS

    html = render_html(_payload(history=_history(60)))
    # One <td> per point, inside the fixed-height sparkline table.
    spark = html.split("Divergence over time")[1].split("</table>")[0]
    assert spark.count("<td") == SPARK_DAYS


def test_the_sparkline_is_omitted_below_two_points():
    """One column is not a trend, and a lone bar at full height reads as one."""
    assert "Divergence over time" not in render_html(_payload(history=_history(1)))


def test_sparkline_columns_scale_to_the_peak_and_never_vanish():
    html = render_html(_payload(history=_history(5)))
    heights = [int(h) for h in re.findall(r"height:(\d+)px;border-radius:2px", html)]
    assert len(heights) == 5
    assert max(heights) == 48                 # the tallest is the peak
    assert min(heights) >= 2                  # and nothing renders as invisible
    assert heights == sorted(heights)         # monotone input, monotone output


def _ratio_row(value: float, foundation: str = "care") -> str:
    """The one row of the diverging chart, isolated from the composition panel
    (which also has a row per foundation)."""
    html = render_html(_payload(comparison={
        "pair": ["self", "modeled_ce"], "log_ratios": {foundation: value}}))
    panel = html.split("Who leans on which foundation")[1]
    return panel.split(f"{foundation}</td>")[1].split("</tr>")[0]


def test_a_positive_log_ratio_lands_on_the_right():
    """The direction the legend promises. Backwards here inverts every claim
    about which diet leans on which foundation."""
    row = _ratio_row(0.5)
    # The left half is an empty fixed-width cell; the bar sits in the right one.
    assert '<td style="width:26%;"></td>' in row
    assert 'align="right"' not in row


def test_a_negative_log_ratio_lands_on_the_left():
    """Left-growing, so both halves extend away from the centre line rather
    than from the page edges."""
    row = _ratio_row(-0.5, "loyalty")
    # The left cell is right-aligned and carries the bar; positive rows have
    # neither (see the test above).
    assert 'align="right"' in row
    assert "border-radius" in row


def test_the_two_sides_of_the_ratio_chart_are_coloured_by_diet():
    from digest.render import DIET_A, DIET_B

    assert DIET_A in _ratio_row(0.5)          # self over-indexes
    assert DIET_B in _ratio_row(-0.5)         # modeled_ce does


# -- one colour per diet, everywhere ----------------------------------------


def test_a_diet_keeps_one_colour_across_every_panel():
    """The bug this replaces: composition coloured by list index, blindspots by
    the own_diet setting. Under the shipped ids (`modeled_ce` sorts before
    `self`) the same diet came out blue in one panel and orange in another."""
    from digest.render import _colours

    payload = _payload(blindspots=[_spot("modeled_ce", "self", "theirs"),
                                   _spot("self", "modeled_ce", "mine")])
    colours = _colours(payload)
    html = render_html(payload, own_diet="self")

    # the composition heading and the theme heading for that diet agree
    for diet_id in ("self", "modeled_ce"):
        heading = html.split(_label(diet_id))[1][:200]
        assert colours[diet_id] in html.split(f"{_label(diet_id)} covered ")[0][-400:] \
            or colours[diet_id] in heading


def test_colours_do_not_depend_on_the_own_diet_setting():
    """Colour means 'which diet', not 'whose blindspot'. An unset own_diet used
    to make every blindspot the same colour."""
    payload = _payload(blindspots=[_spot("modeled_ce", "self", "one"),
                                   _spot("self", "modeled_ce", "two")])
    assert render_html(payload, own_diet=None) == render_html(payload, own_diet="self")


def test_an_unknown_own_diet_is_warned_about(caplog):
    """A typo silently disabled the ordering and said nothing."""
    with caplog.at_level(logging.WARNING):
        build_digest(_payload(), own_diet="slef")
    assert "matches no diet" in caplog.text
    assert "modeled_ce" in caplog.text        # names the ids that do exist


# -- confidence bands -------------------------------------------------------


def _banded():
    payload = _payload()
    payload["diets"][0]["band"] = {
        "care": {"point": 0.28, "low": 0.19, "high": 0.37,
                 "dictionary": 0.31, "transformer": 0.25, "disagreement": 0.06}}
    payload["caveat"] = (
        "Read every number as a noisy estimate, never ground truth. "
        "Whiskers on the radar show the dictionary-vs-transformer range — wider "
        "means the two methods disagree more, so trust that foundation's number less.")
    return payload


def test_the_email_shows_the_ensemble_point_not_the_dictionary_share():
    """The dashboard plots band.point wherever a band exists. Reading the raw
    profile here put a different number on each surface for the same
    foundation, which is exactly the drift the module docstring disclaims."""
    html = render_html(_banded())
    assert "0.280" in html                    # the ensemble point
    assert "0.310" not in html                # not the dictionary-only share


def test_the_band_range_travels_with_the_number():
    """CLAUDE.md §5: ensemble disagreement is the confidence signal. Dropping it
    from the email left the reader with a point estimate and no idea how much to
    trust it."""
    assert "0.19–0.37" in render_html(_banded())
    assert "[0.19-0.37]" in render_text(_banded())


def test_the_caveat_stops_pointing_at_a_radar_that_is_not_there():
    """The dashboard's caveat ends by explaining whiskers. There is no radar in
    an email, so it told the reader to go look at something absent."""
    html = render_html(_banded())
    assert "Whiskers on the radar" not in html
    assert "range printed next to each foundation" in html


# -- the text part is not a stub --------------------------------------------


def test_text_and_html_carry_the_same_sections():
    """The text part is what a screen reader gets. It was missing the sparkline,
    the fairness split, liberty, and liberty's provenance caveat."""
    payload = _payload(
        history=_history(5),
        blindspots=[_spot("modeled_ce", "self")],
        fairness_split={"diets": {"self": {"equality": .71, "proportionality": .29,
                                           "docs_split": 40, "docs_total": 128,
                                           "coverage": .31, "thin": False,
                                           "leans": "equality"}}, "gap": .3},
        liberty={"scorer": "x", "diets": {"self": {"mean": .31, "salient_share": .18,
                                                   "docs_scored": 88, "docs_total": 128,
                                                   "coverage": .69, "thin": False}},
                 "gap": .1})
    html, text = render_html(payload), render_text(payload)
    for section in ("Divergence over time", "equality", "proportionality",
                    "Liberty", "least corroborated", "Summaries",
                    "one-sided", "identical pipeline"):
        assert section.lower() in html.lower(), f"missing from html: {section}"
        assert section.lower() in text.lower(), f"missing from text: {section}"


def test_the_text_log_ratios_name_which_diet_is_which():
    """'positive = first diet leans harder' left 'first' undefined."""
    text = render_text(_payload())
    assert "Self over-indexes" in text


# -- a fairness split with nothing in it ------------------------------------


def test_a_split_with_no_documents_is_not_rendered_as_zero():
    """FairnessProfile returns 0.0/0.0 when there is no fairness mass, so the
    `equality is None` guard never fired and an unsplit diet printed
    'equality 0.00 / proportionality 0.00' — unscored rendered as zero."""
    payload = _payload(fairness_split={
        "diets": {"self": {"equality": 0.0, "proportionality": 0.0, "docs_split": 0,
                           "docs_total": 128, "coverage": 0.0, "thin": True,
                           "leans": None}},
        "gap": None})
    html = render_html(payload).split("Reported separately")[1]
    text = render_text(payload).split("Fairness: equality")[1]
    for rendered in (html, text):
        assert "not enough split-terms" in rendered
        assert "0.00" not in rendered


# -- port parsing -----------------------------------------------------------


def test_an_empty_port_falls_back_instead_of_crashing():
    """.env.example is a file of `KEY=` lines, so blanking the port is the
    natural thing to do. int('') was an uncaught traceback out of the CLI."""
    from digest.send import DEFAULT_PORT

    assert MailConfig.from_env({**ENV, "PARALLAX_SMTP_PORT": ""}).port == DEFAULT_PORT


def test_a_nonsense_port_falls_back_and_says_so(caplog):
    from digest.send import DEFAULT_PORT

    with caplog.at_level(logging.WARNING):
        config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_PORT": "smtp"})
    assert config.port == DEFAULT_PORT
    assert "not a port number" in caplog.text


def test_a_real_port_is_honoured():
    assert MailConfig.from_env({**ENV, "PARALLAX_SMTP_PORT": "465"}).port == 465


# -- the CLI ----------------------------------------------------------------


def test_dry_run_writes_the_html_and_exits_zero(tmp_path, monkeypatch):
    from digest.__main__ import main

    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    out = tmp_path / "brief.html"
    assert main(["--db", ":memory:", "--dry-run", "--out", str(out)]) == 0
    assert "<!DOCTYPE html>" in out.read_text()


def test_dry_run_text_writes_the_plain_part(tmp_path, monkeypatch):
    from digest.__main__ import main

    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    out = tmp_path / "brief.txt"
    assert main(["--db", ":memory:", "--dry-run", "--text", "--out", str(out)]) == 0
    body = out.read_text()
    assert "PARALLAX" in body
    assert "<html>" not in body


def test_dry_run_only_flags_are_rejected_rather_than_ignored(monkeypatch):
    """Silently ignoring --open reads as 'it didn't work'."""
    from digest.__main__ import main

    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    with pytest.raises(SystemExit):
        main(["--db", ":memory:", "--open"])


def test_an_unconfigured_send_exits_nonzero_with_the_reason(tmp_path, monkeypatch, capsys):
    from digest.__main__ import main

    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: _payload())
    assert main(["--db", ":memory:"]) == 1
    assert "not configured" in capsys.readouterr().err


def test_the_password_stays_out_of_the_repr():
    """A dataclass repr is one `logger.debug(config)` away from writing a live
    mail password into a log file — and for a Gmail app password that is full
    mailbox access, not send-only."""
    config = MailConfig.from_env({**ENV, "PARALLAX_SMTP_PASSWORD": "s3cret-app-pw"})
    for rendered in (repr(config), str(config), f"{config}"):
        assert "s3cret-app-pw" not in rendered
    assert config.password == "s3cret-app-pw"      # still usable for login
    assert "smtp.example.com" in repr(config)      # the rest still debuggable
