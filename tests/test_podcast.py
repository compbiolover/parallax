"""Podcast ingestion: feed parsing, the episode ledger, and the budgets.

Driven against a fake transcriber and a local HTTP server rather than
faster-whisper and the network — the logic worth testing is what gets
transcribed and what gets skipped, and both are decided before a model is ever
asked for anything.
"""

from __future__ import annotations

import http.server
import threading
from datetime import UTC, datetime, timedelta

import pytest

from ingestion.datastore import Datastore
from ingestion.podcast import (
    Episode,
    PodcastConfig,
    _audio_enclosure,
    _duration_seconds,
    download_audio,
    parse_podcast_feed,
    recent,
    run,
)


def _feed_xml(items: str) -> str:
    return f"""<?xml version="1.0"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel><title>A Show</title>{items}</channel>
</rss>"""


def _item(guid="ep-1", title="Episode One", audio="{BASE}/1.mp3",
          date="Mon, 03 Aug 2026 10:00:00 +0000", duration="1:02:03", enclosure=True):
    enc = f'<enclosure url="{audio}" type="audio/mpeg" length="1000"/>' if enclosure else ""
    return f"""<item>
      <title>{title}</title><guid>{guid}</guid><pubDate>{date}</pubDate>
      <link>http://example.com/{guid}</link>
      <itunes:duration>{duration}</itunes:duration>{enc}
    </item>"""


@pytest.fixture
def feed_server(tmp_path):
    """Serves a feed and a fake mp3 over real HTTP, on a real socket.

    `feedparser` and `urllib` both take a URL, and stubbing them out would test
    the stubs. A loopback server costs milliseconds and exercises the actual
    fetch path, including the size cap.
    """
    state = {"feed": None, "audio": b"\0" * 2048}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith(".mp3"):
                body, ctype = state["audio"], "audio/mpeg"
            else:
                resolved = state["feed"].replace("{BASE}", state["url"])
                body, ctype = resolved.encode(), "application/rss+xml"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    state["feed"] = _feed_xml(_item())
    yield state
    server.shutdown()


# -- feed parsing -----------------------------------------------------------


def test_duration_accepts_every_itunes_shape():
    assert _duration_seconds({"itunes_duration": "42"}) == 42
    assert _duration_seconds({"itunes_duration": "3:20"}) == 200
    assert _duration_seconds({"itunes_duration": "1:02:03"}) == 3723
    assert _duration_seconds({"itunes_duration": ""}) is None
    assert _duration_seconds({"itunes_duration": "about an hour"}) is None


def test_an_enclosure_without_a_type_is_still_audio():
    """Type is advisory and some feeds omit it. Dropping those would silently
    lose whole shows."""
    assert _audio_enclosure({"enclosures": [{"href": "http://x/1.mp3"}]}) == "http://x/1.mp3"
    assert _audio_enclosure({"enclosures": [{"href": "http://x/1.mp3", "type": "audio/mpeg"}]})
    pdf = {"href": "http://x/a.pdf", "type": "application/pdf"}
    assert _audio_enclosure({"enclosures": [pdf]}) is None
    assert _audio_enclosure({"enclosures": []}) is None


def test_entries_without_audio_are_dropped(feed_server):
    """A show that also posts text to the same feed should not book a
    transcription slot for the text."""
    feed_server["feed"] = _feed_xml(
        _item(guid="urn:show:a") + _item(guid="urn:show:b", enclosure=False))
    episodes = parse_podcast_feed(f"{feed_server['url']}/feed.xml", "test-agent")
    assert [e.guid for e in episodes] == ["urn:show:a"]
    assert episodes[0].duration_seconds == 3723
    assert episodes[0].link == "http://example.com/urn:show:a"


def test_a_relative_guid_is_resolved_against_the_feed_url(feed_server):
    """Documented, not desired: feedparser treats a permalink guid as a URI and
    resolves it, so a feed that moves hosts hands back different guids for the
    same episodes. That is why the ledger also keys on the enclosure URL."""
    feed_server["feed"] = _feed_xml(_item(guid="ep-1"))
    episode = parse_podcast_feed(f"{feed_server['url']}/feed.xml", "test-agent")[0]
    assert episode.guid == f"{feed_server['url']}/ep-1"


def test_the_audio_url_stands_in_for_a_missing_guid(feed_server):
    feed_server["feed"] = _feed_xml(
        '<item><title>No guid</title>'
        '<enclosure url="http://example.com/x.mp3" type="audio/mpeg"/></item>')
    episodes = parse_podcast_feed(f"{feed_server['url']}/feed.xml", "test-agent")
    assert episodes[0].guid == "http://example.com/x.mp3"


# -- the window -------------------------------------------------------------


def _ep(guid, days_ago=None):
    when = None
    if days_ago is not None:
        when = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    return Episode(guid=guid, title=guid, audio_url=f"http://x/{guid}.mp3",
                   published_utc=when)


def test_recent_keeps_the_window_and_anything_undated():
    episodes = [_ep("new", 1), _ep("old", 30), _ep("undated")]
    kept = {e.guid for e in recent(episodes, since_days=7)}
    # Undated is kept on purpose: ingesting nothing from a feed that omits
    # dates is worse than transcribing one old episode, which the ledger then
    # stops happening twice.
    assert kept == {"new", "undated"}


def test_a_zero_window_means_no_window():
    assert len(recent([_ep("old", 900)], since_days=0)) == 1


# -- downloading ------------------------------------------------------------


def test_download_writes_the_file_and_the_caller_removes_it(feed_server):
    import os

    path = download_audio(f"{feed_server['url']}/1.mp3", "test-agent")
    try:
        assert os.path.getsize(path) == 2048
    finally:
        os.unlink(path)


def test_an_oversized_declared_length_is_refused_before_downloading(feed_server):
    """Content-Length first, so an oversized enclosure costs one round trip
    rather than a full download."""
    with pytest.raises(ValueError, match="declares"):
        download_audio(f"{feed_server['url']}/1.mp3", "test-agent", max_bytes=100)


def test_a_refused_download_leaves_no_file_behind(feed_server, tmp_path, monkeypatch):
    import glob
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ValueError):
        download_audio(f"{feed_server['url']}/1.mp3", "test-agent", max_bytes=100)
    assert glob.glob(str(tmp_path / "parallax-*")) == []


# -- the run ----------------------------------------------------------------


class _FakeTranscriber:
    name = "fake-whisper"

    def __init__(self, text="the moral of the story is care for one another " * 20):
        self.text = text
        self.calls: list[str] = []

    def transcribe(self, path):
        self.calls.append(path)
        return self.text


def _registry(url, ingest_type="podcast_rss"):
    from ingestion.config import Persona, Registry, Source, Stratum

    return Registry(
        version=3,
        strata=[Stratum(id="talk_radio")],
        sources=[
            Source(id="show", name="A Show", medium="podcast", role="talk",
                   ingest_type=ingest_type, url=url, stratum_id="talk_radio"),
        ],
        personas=[
            Persona(id="modeled_ce", label="Modeled", family="right",
                    stratum_weights={"talk_radio": 1.0},
                    source_weights={"show": 1.0}),
        ],
    )


def _run(store, feed_server, transcriber, **kwargs):
    return run(store, _registry(f"{feed_server['url']}/feed.xml"),
               transcriber=transcriber, **kwargs)


def test_an_episode_is_transcribed_scored_and_recorded(feed_server):
    store = Datastore(":memory:")
    fake = _FakeTranscriber()
    stats = _run(store, feed_server, fake, podcast_config=PodcastConfig(since_days=0))
    assert stats.transcribed == 1 and stats.stored == 1
    assert len(fake.calls) == 1
    assert store.counts()["documents"] == 1
    assert store.episode_counts() == {"transcribed": 1}
    store.close()


def test_a_second_run_transcribes_nothing(feed_server):
    """The whole point of the ledger. Re-fetching an article is cheap; paying
    for an hour of audio twice is not."""
    store = Datastore(":memory:")
    fake = _FakeTranscriber()
    _run(store, feed_server, fake, podcast_config=PodcastConfig(since_days=0))
    stats = _run(store, feed_server, fake, podcast_config=PodcastConfig(since_days=0))
    assert len(fake.calls) == 1            # not called again
    assert stats.transcribed == 0
    assert stats.already_seen == 1
    store.close()


def test_a_failed_episode_is_recorded_so_it_is_not_retried_forever(feed_server):
    store = Datastore(":memory:")

    class _Boom(_FakeTranscriber):
        def transcribe(self, path):
            raise RuntimeError("model exploded")

    stats = _run(store, feed_server, _Boom(), podcast_config=PodcastConfig(since_days=0))
    assert stats.failed == 1
    assert store.episode_counts() == {"failed": 1}
    # And the next run skips it rather than paying for the same failure daily.
    fake = _FakeTranscriber()
    _run(store, feed_server, fake, podcast_config=PodcastConfig(since_days=0))
    assert fake.calls == []
    store.close()


def test_an_empty_transcript_is_skipped_not_stored(feed_server):
    store = Datastore(":memory:")
    stats = _run(store, feed_server, _FakeTranscriber(text="   "),
                 podcast_config=PodcastConfig(since_days=0))
    assert stats.skipped == 1 and stats.stored == 0
    assert store.episode_counts() == {"skipped": 1}
    store.close()


def test_the_per_source_episode_cap_holds(feed_server):
    feed_server["feed"] = _feed_xml("".join(_item(guid=f"ep-{i}") for i in range(6)))
    store = Datastore(":memory:")
    fake = _FakeTranscriber()
    _run(store, feed_server, fake,
         podcast_config=PodcastConfig(since_days=0, max_episodes_per_source=2))
    assert len(fake.calls) == 2
    store.close()


def test_a_spent_time_budget_stops_the_run_and_says_so(feed_server):
    feed_server["feed"] = _feed_xml("".join(_item(guid=f"ep-{i}") for i in range(3)))
    store = Datastore(":memory:")
    fake = _FakeTranscriber()
    notes: list[str] = []
    _run(store, feed_server, fake,
         podcast_config=PodcastConfig(since_days=0, time_budget_seconds=0),
         progress=notes.append)
    assert fake.calls == []
    # Truncation is never silent: a short run that reports success is how a
    # diet quietly loses half its audio.
    assert any("time budget" in n for n in notes)
    store.close()


def test_audio_is_deleted_even_when_transcription_fails(feed_server, tmp_path, monkeypatch):
    import glob
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    store = Datastore(":memory:")

    class _Boom(_FakeTranscriber):
        def transcribe(self, path):
            raise RuntimeError("model exploded")

    _run(store, feed_server, _Boom(), podcast_config=PodcastConfig(since_days=0))
    assert glob.glob(str(tmp_path / "parallax-*")) == []
    store.close()


def test_rss_sources_are_not_touched(feed_server):
    """This step reads `podcast_rss` only. Article feeds belong to
    `pipeline.run`, which fetches bodies rather than enclosures."""
    store = Datastore(":memory:")
    fake = _FakeTranscriber()
    stats = run(store, _registry(f"{feed_server['url']}/feed.xml", ingest_type="rss"),
                transcriber=fake, podcast_config=PodcastConfig(since_days=0))
    assert fake.calls == []
    assert stats.considered == 0
    store.close()


def test_settings_budgets_are_read():
    cfg = PodcastConfig.from_settings({"ingestion": {"audio": {
        "max_episodes_per_source": 9, "since_days": 30, "time_budget_seconds": 60,
        "max_megabytes": 10, "whisper_model": "large-v3", "filter_silence": False,
    }}})
    assert cfg.max_episodes_per_source == 9
    assert cfg.since_days == 30
    assert cfg.time_budget_seconds == 60
    assert cfg.max_bytes == 10 * 1024 * 1024
    assert cfg.whisper_model == "large-v3"
    assert cfg.vad_filter is False


def test_missing_faster_whisper_degrades_the_run_rather_than_failing(feed_server, caplog):
    """Same contract as the transformer and liberty taggers: an optional
    dependency costs the run its audio, not its articles."""
    import logging

    store = Datastore(":memory:")
    with caplog.at_level(logging.WARNING):
        stats = run(store, _registry(f"{feed_server['url']}/feed.xml"),
                    podcast_config=PodcastConfig(whisper_model="definitely-not-a-model"))
    assert stats.transcribed == 0
    assert "podcast ingestion skipped" in caplog.text
    store.close()


def test_a_moved_feed_does_not_re_transcribe_the_back_catalogue(feed_server):
    """The reason the ledger keys on the enclosure URL as well as the guid.

    A relative guid is resolved against the feed's own URL, so changing hosts —
    or swapping a public feed for a subscriber one — hands back a different guid
    for every episode. Keyed on guid alone, the next run would re-transcribe the
    whole archive."""
    store = Datastore(":memory:")
    fake = _FakeTranscriber()
    audio = f"{feed_server['url']}/1.mp3"

    store.record_episode(guid=f"{feed_server['url']}/old-host/ep-1", source_id="show",
                         status="transcribed", audio_url=audio)
    feed_server["feed"] = _feed_xml(_item(guid="ep-1", audio=audio))
    stats = _run(store, feed_server, fake, podcast_config=PodcastConfig(since_days=0))

    assert fake.calls == []                # recognised by its audio url
    assert stats.already_seen == 1
    store.close()


def test_an_old_store_gains_the_audio_column(tmp_path):
    """Opening a database created before this column upgrades it in place,
    rather than failing on the first episode written."""
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE podcast_episodes (
            guid TEXT PRIMARY KEY, source_id TEXT NOT NULL, title TEXT,
            published_utc TEXT, processed_utc TEXT NOT NULL, status TEXT NOT NULL,
            detail TEXT, document_id TEXT, duration_seconds INTEGER);
    """)
    conn.commit()
    conn.close()

    store = Datastore(str(path))
    store.record_episode(guid="g", source_id="s", status="transcribed",
                         audio_url="http://x/1.mp3")
    assert store.seen_episode_keys("s") == {"g", "http://x/1.mp3"}
    store.close()


def test_two_shows_may_share_a_guid(feed_server):
    """A guid is unique within a feed and nowhere else — `isPermaLink="false"`
    guids like `12` are legal and common. Keyed globally, the second show's
    episode would overwrite the first show's row, leaving one show skipping an
    episode it never transcribed and the other re-transcribing one it did."""
    store = Datastore(":memory:")
    store.record_episode(guid="12", source_id="show-a", status="transcribed",
                         document_id=None, audio_url="http://a/12.mp3")
    store.record_episode(guid="12", source_id="show-b", status="failed",
                         detail="boom", audio_url="http://b/12.mp3")

    assert store.seen_episode_keys("show-a") == {"12", "http://a/12.mp3"}
    assert store.seen_episode_keys("show-b") == {"12", "http://b/12.mp3"}
    assert store.episode_counts("show-a") == {"transcribed": 1}
    assert store.episode_counts("show-b") == {"failed": 1}
    store.close()


def test_a_guid_keyed_store_is_rebuilt_on_the_composite_key(tmp_path):
    """Rows are carried across, not dropped: each one stands for an episode
    already paid for in CPU time."""
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE podcast_episodes (
            guid TEXT PRIMARY KEY, source_id TEXT NOT NULL, title TEXT,
            published_utc TEXT, processed_utc TEXT NOT NULL, status TEXT NOT NULL,
            detail TEXT, document_id TEXT, duration_seconds INTEGER);
        INSERT INTO podcast_episodes (guid, source_id, processed_utc, status)
            VALUES ('ep-1', 'show', '2026-08-01T00:00:00+00:00', 'transcribed');
    """)
    conn.commit()
    conn.close()

    store = Datastore(str(path))
    keys = {r["pk"] for r in store.conn.execute("PRAGMA table_info(podcast_episodes)")}
    assert keys == {0, 1, 2}                      # composite, not single
    assert store.seen_episode_keys("show") == {"ep-1"}   # the row survived
    # And the rebuilt table takes a colliding guid from another show.
    store.record_episode(guid="ep-1", source_id="other", status="failed")
    assert store.episode_counts("show") == {"transcribed": 1}
    store.close()


def test_already_seen_counts_the_window_not_the_archive(feed_server):
    """The number that says whether the ledger is working has to mean that.
    Counted against the whole feed, a show with a long back catalogue reports
    thousands "already seen" every run — which is the window filtering, not the
    ledger skipping."""
    feed_server["feed"] = _feed_xml(
        _item(guid="urn:new", date="Mon, 03 Aug 2026 10:00:00 +0000")
        + _item(guid="urn:old-1", date="Mon, 03 Aug 2020 10:00:00 +0000")
        + _item(guid="urn:old-2", date="Mon, 03 Aug 2019 10:00:00 +0000"))
    store = Datastore(":memory:")
    fake = _FakeTranscriber()

    first = _run(store, feed_server, fake, podcast_config=PodcastConfig(since_days=3650))
    assert first.already_seen == 0
    assert len(fake.calls) == 3

    # A normal window now: only the recent episode is in it, so it is the only
    # one the ledger can be credited with skipping. Counted against the whole
    # feed this said 3, two of which the ledger had nothing to do with.
    second = _run(store, feed_server, fake, podcast_config=PodcastConfig(since_days=7))
    assert second.already_seen == 1
    assert len(fake.calls) == 3          # nothing new transcribed
    store.close()
