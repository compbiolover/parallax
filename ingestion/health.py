"""Is every source still reachable? Read from the per-run ledger ``source_runs``.

A tool like this dies quietly when its sources do: a feed moves, a site starts
refusing the fetcher, a paywall goes up, and the persona that leaned on it drifts
toward whatever is left without anything saying so. This turns the ledger the
RSS run writes into a short list of what needs attention, and into how much of
each persona's weighted diet is affected — the number that says whether a
comparison is still the comparison it claims to be.

Only RSS runs write the ledger. GDELT backfill stores documents under the same
source ids, so document counts cannot tell a dead feed from a live one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

# A source is failing once its most recent runs are all failures. Three, so one
# timeout or one bad morning is not an alarm.
FAILING_RUNS = 3
# Volume drop: the last week stored under this share of the prior weekly average.
DROP_SHARE = 0.25
# ... and only for sources that stored enough before for the ratio to mean anything.
DROP_MIN_WEEKLY = 5.0
LOOKBACK_DAYS = 35

MESSAGES = {
    "unreadable": "feed did not load: check the URL in config/sources.yaml",
    "empty": "feed loads but lists no items",
    "blocked": "feed loads but none of its articles could be read (paywall or blocked)",
}


@dataclass(frozen=True)
class SourceIssue:
    source_id: str
    name: str
    status: str  # 'unreadable' | 'empty' | 'blocked' | 'dropping'
    detail: str
    last_ok: str | None  # date of the last good run in the lookback, if any


def source_issues(store, registry, now: datetime | None = None) -> list[SourceIssue]:
    now = now or datetime.now(UTC)
    since = (now - timedelta(days=LOOKBACK_DAYS)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    runs: dict[str, list] = {}
    for row in store.source_runs_since(since):
        runs.setdefault(row["source_id"], []).append(row)

    issues = []
    for source in registry.ingestable(("rss",)):
        rows = runs.get(source.id)
        if not rows:
            continue  # never run in the lookback: nothing to say yet
        last_ok = next((r["run_utc"][:10] for r in reversed(rows) if r["status"] == "ok"), None)
        recent = rows[-FAILING_RUNS:]
        if len(recent) == FAILING_RUNS and all(r["status"] != "ok" for r in recent):
            status = recent[-1]["status"]
            issues.append(
                SourceIssue(source.id, source.name, status, MESSAGES.get(status, status), last_ok)
            )
            continue
        this_week = sum(r["stored"] for r in rows if r["run_utc"] >= week_ago)
        before = [r for r in rows if r["run_utc"] < week_ago]
        if before:
            span_days = max(7.0, (datetime.fromisoformat(week_ago) - _parse(before[0])).days)
            weekly = sum(r["stored"] for r in before) / (span_days / 7)
            if weekly >= DROP_MIN_WEEKLY and this_week < DROP_SHARE * weekly:
                issues.append(
                    SourceIssue(
                        source.id,
                        source.name,
                        "dropping",
                        f"{this_week} new articles this week, about {weekly:.0f} a week before",
                        last_ok,
                    )
                )
    return issues


def _parse(row) -> datetime:
    return datetime.fromisoformat(row["run_utc"])


def persona_impact(registry, issues: list[SourceIssue]) -> dict[str, float]:
    """Share of each persona's source weight that sits on a source with an issue.

    Only personas with something affected are returned.
    """
    affected = {i.source_id for i in issues}
    impact = {}
    for persona_id in registry.persona_ids():
        weights = registry.weights_for(persona_id)
        total = sum(weights.values())
        lost = sum(w for s, w in weights.items() if s in affected)
        if total and lost:
            impact[persona_id] = lost / total
    return impact


def health_payload(store, registry, now: datetime | None = None) -> dict:
    issues = source_issues(store, registry, now)
    return {
        "issues": [asdict(i) for i in issues],
        "persona_impact": persona_impact(registry, issues),
    }


def format_report(payload: dict) -> str:
    if not payload["issues"]:
        return "All sources with recorded runs are healthy."
    lines = [f"{len(payload['issues'])} source(s) need attention:"]
    for i in payload["issues"]:
        since = f" (last good run {i['last_ok']})" if i["last_ok"] else ""
        lines.append(f"  {i['source_id']:<28} {i['status']:<10} {i['detail']}{since}")
    if payload["persona_impact"]:
        lines.append("\nWeighted share of each persona's diet affected:")
        for persona_id, share in sorted(payload["persona_impact"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {persona_id:<28} {100 * share:.0f}%")
    return "\n".join(lines)
