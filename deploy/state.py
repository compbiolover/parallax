"""Move the run's state between ephemeral container disk and S3, under a lease.

    python -m deploy.state pull      # fetch the database and the lexicon
    python -m deploy.state push      # store the database and the exported payload
    python -m deploy.state acquire   # take the single-writer lease
    python -m deploy.state release   # give it back
    python -m deploy.state check     # resolve everything, touch nothing

Configured entirely by environment, because that is what a task definition can
set and what ``scripts/parallax-entrypoint.sh`` already assumes:

===========================  ==================================================
``PARALLAX_STATE_BUCKET``    Enables all of this. Unset -> every call is a no-op
                             and the run uses local paths, as on a laptop.
``PARALLAX_STATE_PREFIX``    Key prefix inside the bucket. Default ``parallax/``.
``PARALLAX_LEASE_TABLE``     DynamoDB table for the lease. Unset -> no locking.
``PARALLAX_LEASE_ID``        Lease name. Default ``daily``.
``PARALLAX_LEASE_TTL``       Seconds before an abandoned lease can be taken.
                             Default 7200 — longer than a slow ingest, since the
                             failure this guards is a *crashed* holder, and
                             breaking a live one is much worse than waiting.
===========================  ==================================================

Two decisions worth stating, because both are easy to get subtly wrong:

**The lease expiry is enforced in the condition expression, not by DynamoDB's
TTL.** TTL deletion is best-effort and documented as taking up to 48 hours, so a
lease whose TTL has passed may still be sitting there. The TTL attribute is set
anyway, to sweep rows nobody will ever look at again, but correctness comes from
comparing ``expires_at`` inside the conditional write.

**Release is conditional on ownership.** A process that lost its lease to an
expiry — because it hung past the TTL — must not delete the lease its successor
now holds. The fencing token makes that a no-op rather than a silent theft.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ingestion.config import dictionary_lexicon_path, load_settings

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "parallax/"
DEFAULT_LEASE_ID = "daily"
DEFAULT_LEASE_TTL = 7200

# Where each file lives inside the bucket, relative to the prefix.
DB_KEY = "state/parallax.sqlite"
LEXICON_KEY = "lexicon/emfd_scoring.csv"
PAYLOAD_KEYS = ("exports/latest.js", "exports/catalog.js")


class StateError(RuntimeError):
    """Raised for a condition the caller can act on, with the reason in the text."""


def _boto3():
    """Imported here rather than at module scope, like every other heavy dep.

    Both images do install the ``aws`` extra, so boto3 is present in a container.
    The lazy import is for everywhere else: a workstation has no reason to carry
    it, and this module still has to be *importable* there — the entrypoint runs
    ``deploy.state check`` and the no-bucket paths of ``pull`` and ``push`` on
    every start, including runs that never touch S3 at all. A module-scope import
    would turn "no cloud account" into an ImportError at container start.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise StateError(
            "boto3 is not installed. It lives in the 'aws' extra: pip install -e \".[aws]\""
        ) from exc
    return boto3


@dataclass(frozen=True)
class StateConfig:
    """Resolved configuration. ``bucket`` of ``None`` means "do nothing"."""

    bucket: str | None
    prefix: str
    lease_table: str | None
    lease_id: str
    lease_ttl: int

    @property
    def enabled(self) -> bool:
        return self.bucket is not None

    @property
    def locking(self) -> bool:
        return self.enabled and self.lease_table is not None

    def key(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> StateConfig:
        env = os.environ if env is None else env
        prefix = env.get("PARALLAX_STATE_PREFIX", DEFAULT_PREFIX)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        raw_ttl = env.get("PARALLAX_LEASE_TTL", "")
        try:
            ttl = int(raw_ttl) if raw_ttl else DEFAULT_LEASE_TTL
        except ValueError as exc:
            raise StateError(f"PARALLAX_LEASE_TTL is not a number: {raw_ttl!r}") from exc
        if ttl <= 0:
            raise StateError(f"PARALLAX_LEASE_TTL must be positive, got {ttl}")
        return cls(
            bucket=env.get("PARALLAX_STATE_BUCKET") or None,
            prefix=prefix,
            lease_table=env.get("PARALLAX_LEASE_TABLE") or None,
            lease_id=env.get("PARALLAX_LEASE_ID") or DEFAULT_LEASE_ID,
            lease_ttl=ttl,
        )


# -- the lease ---------------------------------------------------------------


def acquire(cfg: StateConfig, *, token: str | None = None, now: float | None = None) -> str:
    """Take the single-writer lease, returning the fencing token.

    Raises :class:`StateError` if someone else holds it and has not expired.
    Deliberately does not retry or block: the scheduled path is sequenced by the
    state machine, so a live lease here means a *second* run is starting, and
    waiting for it would only queue a collision rather than avoid one.
    """
    if not cfg.locking:
        return ""
    token = token or uuid.uuid4().hex
    now = time.time() if now is None else now
    expires = int(now + cfg.lease_ttl)

    table = _boto3().resource("dynamodb").Table(cfg.lease_table)
    try:
        table.put_item(
            Item={
                "lock_id": cfg.lease_id,
                "owner": token,
                "acquired_at": int(now),
                "expires_at": expires,
            },
            # Free, or expired. `expires_at` is compared here rather than trusted
            # to DynamoDB's TTL sweeper, which is best-effort and can lag by
            # many hours.
            ConditionExpression=("attribute_not_exists(lock_id) OR expires_at < :now"),
            ExpressionAttributeValues={":now": int(now)},
        )
    except Exception as exc:
        if type(exc).__name__ != "ConditionalCheckFailedException":
            raise
        held = _describe_lease(cfg)
        raise StateError(
            f"the {cfg.lease_id!r} lease is held by {held.get('owner', 'someone')} "
            f"until {held.get('expires_at', 'unknown')} (epoch seconds). Another run "
            "is in progress; this one is stopping rather than opening the same "
            "SQLite file alongside it."
        ) from exc
    logger.info("lease %s acquired until %s", cfg.lease_id, expires)
    return token


def release(cfg: StateConfig, token: str) -> bool:
    """Give the lease back. Returns False if we no longer owned it.

    Not an error: a run that overran its TTL legitimately lost the lease, and the
    right response is to say so rather than to delete whatever the successor put
    there.
    """
    if not cfg.locking or not token:
        return True
    table = _boto3().resource("dynamodb").Table(cfg.lease_table)
    try:
        table.delete_item(
            Key={"lock_id": cfg.lease_id},
            # `owner` is a DynamoDB reserved word, so it has to be aliased. Used
            # bare, every release fails with a ValidationException and the lease
            # lingers until its TTL — locking the next run out for two hours for
            # no reason, and only on the path nobody exercises by hand.
            ConditionExpression="#owner = :token",
            ExpressionAttributeNames={"#owner": "owner"},
            ExpressionAttributeValues={":token": token},
        )
    except Exception as exc:
        if type(exc).__name__ != "ConditionalCheckFailedException":
            raise
        logger.warning(
            "lease %s was no longer ours to release — it expired and someone else "
            "took it. Leaving it alone.",
            cfg.lease_id,
        )
        return False
    logger.info("lease %s released", cfg.lease_id)
    return True


def _describe_lease(cfg: StateConfig) -> dict:
    table = _boto3().resource("dynamodb").Table(cfg.lease_table)
    got = table.get_item(Key={"lock_id": cfg.lease_id})
    return got.get("Item") or {}


# -- the files ---------------------------------------------------------------


def _download(client, bucket: str, key: str, dest: Path) -> bool:
    """Fetch one object. False when it is simply not there yet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(dest))
    except Exception as exc:
        name = type(exc).__name__
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if name in ("ClientError", "S3UploadFailedError") and code in ("404", "NoSuchKey"):
            return False
        if name == "ClientError" and code == "403":
            raise StateError(
                f"denied reading s3://{bucket}/{key}. The task role needs "
                "s3:GetObject on this prefix."
            ) from exc
        raise
    return True


def pull(cfg: StateConfig, settings: dict | None = None) -> dict[str, str]:
    """Fetch the database and the lexicon. Returns what happened, per file.

    A missing database is normal — it is the first run. A missing *lexicon* is
    not fatal here either, because `--only export` never loads one; the
    entrypoint's ``PARALLAX_REQUIRE_LEXICON`` guard is what refuses to score
    without it, and that check belongs at the point of use rather than here.
    """
    if not cfg.enabled:
        return {}
    settings = load_settings() if settings is None else settings
    from ingestion.config import datastore_path

    client = _boto3().client("s3")
    result: dict[str, str] = {}

    db = Path(datastore_path(settings))
    result["database"] = (
        "downloaded" if _download(client, cfg.bucket, cfg.key(DB_KEY), db) else "absent (first run)"
    )

    lexicon = dictionary_lexicon_path(settings)
    if lexicon:
        found = _download(client, cfg.bucket, cfg.key(LEXICON_KEY), Path(lexicon))
        result["lexicon"] = "downloaded" if found else "absent"
    else:
        result["lexicon"] = "not configured"

    for name, state in result.items():
        logger.info("pull %-9s %s", name, state)
    return result


def push(cfg: StateConfig, settings: dict | None = None, *, payload: bool = True) -> dict[str, str]:
    """Store the database, and optionally the exported payload.

    The database goes up even after a partly failed run. SQLite is
    transactionally consistent whatever happened above it, and throwing away a
    successful ingest because a later step 503'd would lose real work to protect
    nothing. The payload is the opposite: it is only meaningful if `export`
    actually ran, so the caller passes ``payload=False`` when it did not.
    """
    if not cfg.enabled:
        return {}
    settings = load_settings() if settings is None else settings
    from ingestion.config import datastore_path

    client = _boto3().client("s3")
    result: dict[str, str] = {}

    db = Path(datastore_path(settings))
    if db.exists():
        client.upload_file(str(db), cfg.bucket, cfg.key(DB_KEY))
        result["database"] = f"uploaded ({db.stat().st_size} bytes)"
    else:
        # Nothing ran, or it ran somewhere else. Uploading nothing is right;
        # silently reporting success is not.
        result["database"] = "missing locally, not uploaded"

    if payload:
        from dashboard.export import DEFAULT_CATALOG_OUT, DEFAULT_OUT

        for local, key in zip((DEFAULT_OUT, DEFAULT_CATALOG_OUT), PAYLOAD_KEYS, strict=True):
            if Path(local).exists():
                client.upload_file(str(local), cfg.bucket, cfg.key(key))
                result[Path(local).name] = "uploaded"
            else:
                result[Path(local).name] = "missing locally, not uploaded"

    for name, state in result.items():
        logger.info("push %-12s %s", name, state)
    return result


# -- CLI ---------------------------------------------------------------------


def _describe(cfg: StateConfig) -> str:
    if not cfg.enabled:
        return "PARALLAX_STATE_BUCKET is unset — state stays on local disk."
    lines = [
        f"bucket       s3://{cfg.bucket}/{cfg.prefix}",
        f"database     {cfg.key(DB_KEY)}",
        f"lexicon      {cfg.key(LEXICON_KEY)}",
    ]
    lines.append(
        f"lease        {cfg.lease_table}:{cfg.lease_id} (ttl {cfg.lease_ttl}s)"
        if cfg.locking
        else "lease        none — PARALLAX_LEASE_TABLE is unset, so nothing stops "
        "two runs opening the store at once"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deploy.state", description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("pull", "push", "acquire", "release", "check"))
    parser.add_argument("--token", help="fencing token, for release")
    parser.add_argument("--no-payload", action="store_true", help="push the database only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = StateConfig.from_env()

    if args.action == "check":
        print(_describe(cfg))
        return 0
    if not cfg.enabled:
        logger.info("PARALLAX_STATE_BUCKET unset — nothing to do")
        return 0

    try:
        if args.action == "pull":
            pull(cfg)
        elif args.action == "push":
            push(cfg, payload=not args.no_payload)
        elif args.action == "acquire":
            # stdout is the token, so a shell can capture it; logs go to stderr.
            print(acquire(cfg))
        elif args.action == "release":
            if not args.token:
                raise StateError("release needs --token")
            release(cfg, args.token)
    except StateError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
