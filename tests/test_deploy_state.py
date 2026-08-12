"""The lease and the file round-trip, against a simulated S3 and DynamoDB.

The lease is the part worth testing properly. Everything else here is copying
files, where a mistake announces itself; a lock that quietly hands the same
database to two writers announces itself as a corrupted store weeks later, and
`Datastore.__init__` runs a schema script and a migration pass on every open, so
"two writers" is not a mild condition.

The three cases that matter are all here: a second acquirer is refused, an
*expired* lease can be taken, and releasing a lease you no longer own does not
delete the one your successor is holding.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from botocore.exceptions import ClientError  # noqa: E402
from moto import mock_aws  # noqa: E402

from deploy.state import (  # noqa: E402
    DB_KEY,
    LEXICON_KEY,
    StateConfig,
    StateError,
    acquire,
    pull,
    push,
    release,
)

REGION = "us-east-1"
BUCKET = "parallax-test"
TABLE = "parallax-lease-test"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    """moto refuses to run against real credentials, which is the point."""
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SECURITY_TOKEN": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": REGION,
    }.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def aws():
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        ddb = boto3.client("dynamodb", region_name=REGION)
        ddb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "lock_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "lock_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield s3


def _cfg(**over) -> StateConfig:
    base = {
        "bucket": BUCKET,
        "prefix": "parallax/",
        "lease_table": TABLE,
        "lease_id": "daily",
        "lease_ttl": 3600,
    }
    base.update(over)
    return StateConfig(**base)


# -- configuration -----------------------------------------------------------


def test_without_a_bucket_everything_is_a_no_op():
    """The image has to keep working with no AWS at all."""
    cfg = StateConfig.from_env({})
    assert not cfg.enabled
    assert pull(cfg) == {}
    assert push(cfg) == {}
    assert acquire(cfg) == ""
    assert release(cfg, "anything") is True


def test_prefix_gets_a_trailing_slash():
    cfg = StateConfig.from_env({"PARALLAX_STATE_BUCKET": "b", "PARALLAX_STATE_PREFIX": "x"})
    assert cfg.key(DB_KEY) == "x/state/parallax.sqlite"


def test_a_nonsense_ttl_is_rejected_rather_than_silently_defaulted():
    with pytest.raises(StateError, match="not a number"):
        StateConfig.from_env({"PARALLAX_STATE_BUCKET": "b", "PARALLAX_LEASE_TTL": "soon"})
    with pytest.raises(StateError, match="must be positive"):
        StateConfig.from_env({"PARALLAX_STATE_BUCKET": "b", "PARALLAX_LEASE_TTL": "0"})


# -- the lease ---------------------------------------------------------------


def test_a_second_run_is_refused_while_the_lease_is_held(aws):
    cfg = _cfg()
    token = acquire(cfg)
    assert token

    with pytest.raises(StateError, match="lease is held"):
        acquire(cfg)

    assert release(cfg, token) is True
    assert acquire(cfg)  # free again


def test_an_expired_lease_can_be_taken(aws):
    """A crashed holder must not lock the pipeline out forever."""
    cfg = _cfg(lease_ttl=60)
    first = acquire(cfg, now=1_000_000)

    # Same table, an hour and a half later: the first lease has expired.
    second = acquire(cfg, now=1_000_000 + 5400)
    assert second != first


def test_expiry_is_enforced_by_the_write_not_by_dynamodb_ttl(aws):
    """The condition compares expires_at itself.

    DynamoDB's TTL sweeper is best-effort and documented as taking up to 48
    hours, so an expired row is routinely still present. If correctness relied on
    the row being gone, this would deadlock for two days.
    """
    cfg = _cfg(lease_ttl=60)
    acquire(cfg, now=1_000_000)

    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)
    assert table.get_item(Key={"lock_id": "daily"})["Item"], "row should still be there"

    assert acquire(cfg, now=1_000_000 + 999)  # taken anyway


def test_releasing_a_lease_you_lost_does_not_steal_it_back(aws):
    """The fencing token's whole job.

    A run that overran its TTL legitimately lost the lease. If its release
    deleted the row, the successor would be holding a lock that no longer
    exists — and a third run could then start alongside it.
    """
    cfg = _cfg(lease_ttl=60)
    stale = acquire(cfg, now=1_000_000)
    successor = acquire(cfg, now=1_000_000 + 5400)

    assert release(cfg, stale) is False, "should report the loss rather than delete"

    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)
    still_there = table.get_item(Key={"lock_id": "daily"})["Item"]
    assert still_there["owner"] == successor, "the successor's lease survived"


def test_no_lease_table_means_no_locking(aws):
    cfg = _cfg(lease_table=None)
    assert acquire(cfg) == ""
    assert acquire(cfg) == ""  # and again, because nothing is locking


# -- the files ---------------------------------------------------------------


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Point the datastore and lexicon at a temp dir."""
    db = tmp_path / "data" / "parallax.sqlite"
    lex = tmp_path / "data" / "emfd.csv"
    monkeypatch.setattr("ingestion.config.datastore_path", lambda *a, **k: str(db), raising=True)
    monkeypatch.setattr(
        "deploy.state.dictionary_lexicon_path", lambda *a, **k: str(lex), raising=True
    )
    return {"_db": db, "_lex": lex}


def test_a_missing_database_is_a_first_run_not_an_error(aws, settings):
    result = pull(_cfg(), settings)
    assert result["database"] == "absent (first run)"
    assert not settings["_db"].exists()


def test_round_trip_preserves_bytes(aws, settings):
    db: Path = settings["_db"]
    db.parent.mkdir(parents=True, exist_ok=True)
    original = b"SQLite format 3\x00" + bytes(range(256)) * 4
    db.write_bytes(original)

    push(_cfg(), settings, payload=False)
    db.unlink()

    assert pull(_cfg(), settings)["database"] == "downloaded"
    assert db.read_bytes() == original


def test_the_lexicon_comes_down_too(aws, settings):
    lex: Path = settings["_lex"]
    aws.put_object(Bucket=BUCKET, Key=f"parallax/{LEXICON_KEY}", Body=b"word,care_p\na,0.5\n")

    assert pull(_cfg(), settings)["lexicon"] == "downloaded"
    assert lex.read_text().startswith("word,care_p")


def test_pushing_without_a_local_database_reports_it(aws, settings):
    """Better than reporting success for having uploaded nothing."""
    result = push(_cfg(), settings, payload=False)
    assert result["database"] == "missing locally, not uploaded"
    # Specifically a 404, not merely "something raised" — a blind assertion here
    # would also pass on a credentials or region error and prove nothing about
    # whether the object was uploaded.
    with pytest.raises(ClientError) as exc:
        aws.head_object(Bucket=BUCKET, Key=f"parallax/{DB_KEY}")
    assert exc.value.response["Error"]["Code"] == "404"


def test_push_skips_the_payload_when_asked(aws, settings, monkeypatch, tmp_path):
    """`export` did not run, so whatever is on disk describes an older corpus."""
    db: Path = settings["_db"]
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"x")

    result = push(_cfg(), settings, payload=False)
    assert "latest.js" not in result
    assert result["database"].startswith("uploaded")


def test_the_key_layout_is_what_terraform_will_grant(aws, settings):
    """These strings end up in an IAM policy, so pin them."""
    cfg = _cfg()
    assert cfg.key(DB_KEY) == "parallax/state/parallax.sqlite"
    assert cfg.key(LEXICON_KEY) == "parallax/lexicon/emfd_scoring.csv"


def test_boto3_is_only_needed_when_a_bucket_is_configured(monkeypatch):
    """The core image carries no boto3; an unset bucket must not import it."""
    monkeypatch.setitem(os.environ, "PARALLAX_STATE_BUCKET", "")
    cfg = StateConfig.from_env(dict(os.environ))
    monkeypatch.setattr(
        "deploy.state._boto3",
        lambda: (_ for _ in ()).throw(AssertionError("boto3 must not be imported")),
    )
    assert pull(cfg) == {}
    assert push(cfg) == {}
