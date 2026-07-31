"""Group blindspot clusters into themes a person can scan.

The blindspot engine finds coverage asymmetry at the level of a *cluster*, and
a cluster is small on purpose — HDBSCAN in ``leaf`` mode splits the news finely
so that "one cluster" really is "one story". That is the right unit to measure
and the wrong unit to read: a day produces a couple of dozen of them, each
labeled with whatever terms c-TF-IDF found distinctive, and the reader gets a
list too long to finish titled in a vocabulary nobody speaks.

A **theme** is the reading unit. It collects the blindspot clusters that point
the same direction (one diet covered them, the other did not) and belong to the
same subject, under a name written for a human: "Faith & the church", not
"christianity today · christianity · today".

Two ways to get there, in order of preference:

1. **Claude**, given the headlines and the vocabulary below, naming and
   assigning each cluster. Better on subjects the taxonomy has never heard of.
2. **The taxonomy** in :data:`TAXONOMY` — a keyword map, versioned in code,
   which always runs and needs no key. It is also the fallback for any single
   cluster Claude declines to place.

The taxonomy's category names are the reader-facing copy, so they follow the §0
tone rule: they describe subjects, not the people who care about them. Neither
direction of asymmetry gets a name the other would not accept.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_THEME_MODEL = "claude-sonnet-5"

OTHER_KEY = "other"
OTHER_TITLE = "Other coverage"

# (key, human title, keywords). Order is the tie-break: a cluster matching two
# themes equally takes the earlier one, so specific subjects precede the
# catch-alls ("politics", "other") they would otherwise be swallowed by.
TAXONOMY: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("life", "Abortion & bioethics", (
        "abortion", "pro-life", "prolife", "roe", "planned parenthood",
        "euthanasia", "assisted suicide", "embryo", "embryos", "fetal", "fetus",
        "bioethics", "ivf", "surrogacy",
    )),
    ("faith", "Faith & the church", (
        "church", "churches", "pastor", "pastors", "ministry", "ministries",
        "congregation", "congregations", "evangelical", "evangelicals",
        "christian", "christians", "christianity", "catholic", "catholics",
        "pope", "vatican", "bible", "biblical", "scripture", "gospel", "prayer",
        "worship", "missionary", "missionaries", "missions", "seminary",
        "theology", "theological", "faith", "religious", "religion",
        "denomination", "baptist", "presbyterian", "methodist", "lutheran",
        "anglican", "synagogue", "mosque", "jewish", "muslim", "islam",
        "persecution", "revival", "discipleship", "sermon", "parish", "diocese",
        "clergy", "chaplain", "sacred", "holy", "sin", "repentance",
    )),
    ("family", "Family, marriage & sexuality", (
        "marriage", "married", "divorce", "family", "families", "parents",
        "parenting", "motherhood", "fatherhood", "mom", "dad", "adoption",
        "foster", "children", "kids", "birth", "newborn", "pregnancy",
        "pregnant", "fertility", "gender", "transgender", "lgbt", "lgbtq",
        "sexuality", "dating", "singleness",
    )),
    ("immigration", "Immigration & the border", (
        "immigration", "immigrant", "immigrants", "migrant", "migrants",
        "border", "deportation", "deported", "asylum", "refugee", "refugees",
        "visa", "visas", "citizenship", "naturalization",
    )),
    ("elections", "Elections & campaigns", (
        "election", "elections", "voter", "voters", "voting", "ballot",
        "ballots", "primary", "caucus", "campaign", "poll", "polls", "polling",
        "candidate", "candidates", "midterm", "midterms", "turnout",
        "gerrymander", "recount",
    )),
    ("crime", "Crime & the courts", (
        "arrest", "arrested", "charged", "charges", "indicted", "indictment",
        "murder", "homicide", "shooting", "stabbing", "assault", "rape",
        "trafficking", "trial", "verdict", "guilty", "acquitted", "plea",
        "sentenced", "sentencing", "prison", "jail", "inmate", "police",
        "officer", "sheriff", "deputy", "lawsuit", "sued", "judge", "court",
        "courts", "appeals", "prosecutor", "prosecutors", "fraud", "theft",
        "stole", "steals", "robbery", "burglary", "dui", "manhunt", "suspect",
        "felony", "bodycam", "custody",
    )),
    ("foreign", "War & foreign affairs", (
        "war", "ukraine", "russia", "russian", "israel", "israeli", "gaza",
        "hamas", "hezbollah", "iran", "iranian", "china", "chinese", "taiwan",
        "nato", "troops", "military", "airstrike", "airstrikes", "ceasefire",
        "hostage", "hostages", "missile", "missiles", "diplomacy", "diplomat",
        "sanctions", "treaty", "embassy", "north korea", "venezuela", "syria",
        "afghanistan", "nigeria", "sudan", "united nations", "foreign",
    )),
    ("disaster", "Disasters & public safety", (
        "hurricane", "tornado", "wildfire", "wildfires", "flood", "flooding",
        "earthquake", "quake", "storm", "evacuation", "evacuated", "crash",
        "collapse", "rescue", "rescued", "drought", "blizzard", "landslide",
        "derailment",
    )),
    ("health", "Health & medicine", (
        "health", "hospital", "hospitals", "doctor", "doctors", "nurse",
        "patient", "patients", "cancer", "disease", "virus", "vaccine",
        "vaccines", "covid", "measles", "outbreak", "epidemic", "addiction",
        "overdose", "opioid", "fentanyl", "therapy", "surgery", "surgical",
        "kidney", "heart", "diabetes", "obesity", "diagnosis", "treatment",
        "medicaid", "medicare", "fda", "cdc", "mental health", "suicide",
    )),
    ("education", "Schools & universities", (
        "school", "schools", "student", "students", "teacher", "teachers",
        "university", "universities", "college", "colleges", "campus",
        "curriculum", "homeschool", "homeschooling", "tuition", "professor",
        "classroom", "education", "kindergarten", "graduation",
    )),
    ("economy", "Economy, work & prices", (
        "economy", "economic", "inflation", "prices", "jobs", "unemployment",
        "wages", "tariff", "tariffs", "trade", "stocks", "market", "markets",
        "recession", "housing", "mortgage", "rent", "layoffs", "union",
        "workers", "taxes", "earnings", "bankruptcy", "debt", "budget deficit",
        "interest rates", "federal reserve",
    )),
    ("climate", "Climate, energy & environment", (
        "climate", "emissions", "carbon", "warming", "solar", "renewable",
        "energy", "oil", "pipeline", "drilling", "pollution", "conservation",
        "wildlife", "endangered", "recycling", "epa",
    )),
    ("science", "Science & technology", (
        "artificial intelligence", "robot", "robotics", "nasa", "space",
        "satellite", "spacecraft", "research", "researchers", "scientists",
        "scientific", "quantum", "chip", "chips", "software", "algorithm",
        "engineers", "physics", "genome", "dinosaur", "telescope", "exosuit",
        "technology", "startup", "chatbot",
    )),
    ("sports", "Sports", (
        "nfl", "nba", "mlb", "wnba", "ncaa", "football", "basketball",
        "baseball", "soccer", "hockey", "olympics", "olympic", "coach",
        "quarterback", "playoff", "playoffs", "touchdown", "championship",
        "golf", "tennis", "athlete", "athletes", "roster", "draft pick",
    )),
    ("culture", "Culture & entertainment", (
        "movie", "movies", "film", "music", "album", "song", "singer", "actor",
        "actress", "celebrity", "concert", "tour", "book", "novel", "award",
        "awards", "grammy", "oscar", "festival", "series", "streaming",
        "hollywood", "podcast",
    )),
    ("media", "Media & speech", (
        "media", "journalist", "journalists", "journalism", "newsroom",
        "censorship", "free speech", "first amendment", "social media",
        "twitter", "facebook", "tiktok", "youtube", "misinformation",
        "broadcast", "anchor",
    )),
    ("politics", "Politics & government", (
        "congress", "senate", "senator", "house", "lawmakers", "legislature",
        "bill", "legislation", "governor", "mayor", "white house", "president",
        "presidential", "administration", "federal", "agency", "regulation",
        "shutdown", "veto", "hearing", "subpoena", "impeachment", "policy",
        "republican", "republicans", "democrat", "democrats", "gop",
        "conservative", "liberal", "supreme court", "constitution",
    )),
)

_COMPILED: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (
        key,
        title,
        re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b"),
    )
    for key, title, keywords in TAXONOMY
)

_TITLES = {key: title for key, title, _ in TAXONOMY} | {OTHER_KEY: OTHER_TITLE}


@dataclass(frozen=True)
class ThemeAssignment:
    """Which theme one cluster belongs to, and who decided."""

    key: str
    title: str
    method: str  # 'taxonomy' | 'claude'


@dataclass
class Theme:
    """One direction of asymmetry on one subject — the email's card."""

    key: str
    title: str
    dominant_diet: str
    other_diet: str
    cluster_count: int
    story_count: int
    one_sided: float
    stories: list[str] = field(default_factory=list)
    method: str = "taxonomy"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "dominant_diet": self.dominant_diet,
            "other_diet": self.other_diet,
            "cluster_count": self.cluster_count,
            "story_count": self.story_count,
            "one_sided": self.one_sided,
            "stories": list(self.stories),
            "method": self.method,
        }


# -- taxonomy ---------------------------------------------------------------


def taxonomy_theme(titles: list[str]) -> ThemeAssignment:
    """Best-matching taxonomy entry for a cluster's headlines.

    Ranked on *reach* first — how many of the cluster's headlines a theme
    touches at all — then on specificity, and only then on raw keyword hits.
    Hits alone hand almost every cluster to the broadest categories: "Senate
    Republicans weigh an abortion bill" is one abortion word against three of
    politics', and the story is not about the Senate. Reach asks the question
    that distinguishes them — does this subject describe the whole cluster, or
    one clause of one headline? — and taxonomy order settles the rest, which is
    why the catch-alls are written last.

    No hits at all is an honest answer, not a failure: those clusters collect
    under "Other coverage" rather than being forced into a category.
    """
    reach: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for title in titles:
        lowered = title.lower()
        for key, _, pattern in _COMPILED:
            matched = len({m.group(0) for m in pattern.finditer(lowered)})
            if matched:
                reach[key] += 1
                hits[key] += matched
    if not reach:
        return ThemeAssignment(OTHER_KEY, OTHER_TITLE, "taxonomy")
    best = max(reach, key=lambda k: (reach[k], -_order(k), hits[k]))
    return ThemeAssignment(best, _TITLES[best], "taxonomy")


def _order(key: str) -> int:
    for i, (k, _, _) in enumerate(TAXONOMY):
        if k == key:
            return i
    return len(TAXONOMY)


# -- Claude -----------------------------------------------------------------


_SYSTEM = """You group news clusters into topical themes for a tool that \
compares two media diets through Moral Foundations Theory.

Rules:
- A theme names a SUBJECT, never the people who follow it. "Faith & the church", \
not "religious conservatives". Never characterize either diet or its audience.
- Titles are 2-5 words and at most 42 characters, suitable as a card heading a \
person reads on a phone. Plain text: letters, digits, spaces, and & ' / , . - \
are the only characters allowed, so "Israel-Hamas war" and "Faith, family & \
work" are fine and anything with markup or quotes in it is not.
- Prefer a key from the supplied vocabulary when the cluster fits it. Invent a \
new key only when nothing fits, and keep it lowercase with hyphens.
- Group aggressively: 3-6 themes total across all clusters is the target. A \
theme holding one cluster should be rare.
- Output JSON only."""

# Kept in step with the character list in _SYSTEM above, in both directions: a
# rule the prompt states and the validator does not enforce is a rule that is
# not there, and one the validator enforces without stating drops good titles
# for a reason the model was never told.
_MAX_TITLE_CHARS = 42
_TITLE_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 &'/,.-]*$")
_KEY_OK = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")


def claude_assignments(
    entries: list[tuple[int, list[str]]],
    client: object | None = None,
    model: str = DEFAULT_THEME_MODEL,
) -> dict[int, ThemeAssignment]:
    """Ask Claude to theme each cluster. ``{}`` when it cannot or will not.

    Every failure path returns ``{}`` rather than raising: the taxonomy is
    always there behind this, and a themed brief that is one day less clever
    beats no brief. Anything Claude returns that does not validate — a missing
    cluster, a key that is not a key, a title long enough to break the card — is
    dropped for that cluster alone, which then falls back on its own.
    """
    if not entries:
        return {}
    if client is None:
        from scoring.claude_client import build_client

        client, reason = build_client()
        if client is None:
            logger.info("Blindspot themes fall back to the taxonomy: %s", reason)
            return {}
    try:
        text = _call(client, model, entries)
        raw = _parse(text)
    except Exception as exc:  # network, quota, malformed JSON — all the same here
        logger.warning("Claude theming failed, using the taxonomy: %s", exc)
        return {}

    ids = {cid for cid, _ in entries}
    out: dict[int, ThemeAssignment] = {}
    for item in raw:
        try:
            cid = int(item["cluster_id"])
            key = str(item["key"]).strip().lower()
            title = " ".join(str(item["title"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        if cid not in ids or cid in out:
            continue
        if not _KEY_OK.match(key) or not _TITLE_OK.match(title):
            continue
        if len(title) > _MAX_TITLE_CHARS:
            continue
        out[cid] = ThemeAssignment(key, title, "claude")
    return out


def _call(client, model: str, entries: list[tuple[int, list[str]]]) -> str:
    vocabulary = ", ".join(f"{key} ({title})" for key, title, _ in TAXONOMY)
    lines = [
        "Assign every cluster below to a theme.",
        "",
        f"Vocabulary of preferred keys: {vocabulary}, {OTHER_KEY} ({OTHER_TITLE}).",
        "",
        'Respond with JSON only: {"assignments": [{"cluster_id": 0, "key": '
        '"faith", "title": "Faith & the church"}]}. Every cluster id below must '
        "appear exactly once. Clusters sharing a key must share the same title.",
        "",
    ]
    for cid, titles in entries:
        lines.append(f"cluster {cid}:")
        lines.extend(f"  - {t}" for t in titles[:6])
    resp = client.messages.create(
        model=model,
        # One assignment object runs ~40 tokens. A fixed ceiling is fine until
        # the day the corpus is big enough to need more of them, and that is
        # the day it truncates mid-JSON and the whole batch silently falls back
        # to the taxonomy — the failure that looks like nothing happening.
        max_tokens=min(8000, 400 + 60 * len(entries)),
        system=_SYSTEM,
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


def _parse(text: str) -> list[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the response")
    data = json.loads(text[start:end + 1])
    assignments = data.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("no 'assignments' list in the response")
    return [a for a in assignments if isinstance(a, dict)]


def assign_themes(
    entries: list[tuple[int, list[str]]],
    client: object | None = None,
    model: str = DEFAULT_THEME_MODEL,
    use_claude: bool = True,
) -> dict[int, ThemeAssignment]:
    """Theme every cluster: Claude where it answered, the taxonomy everywhere else."""
    from_claude = claude_assignments(entries, client, model) if use_claude else {}
    # A key Claude used keeps Claude's wording everywhere, so two clusters it
    # placed together do not arrive under two spellings of one theme.
    titles = {a.key: a.title for a in from_claude.values()}
    out: dict[int, ThemeAssignment] = {}
    for cid, cluster_titles in entries:
        assignment = from_claude.get(cid) or taxonomy_theme(cluster_titles)
        title = titles.get(assignment.key, assignment.title)
        out[cid] = ThemeAssignment(assignment.key, title, assignment.method)
    return out


# -- grouping ---------------------------------------------------------------


@dataclass(frozen=True)
class _Spot:
    """One blindspot, however the caller happens to be holding it.

    :func:`group_blindspots` is called from the cluster run with
    ``Blindspot`` objects and from the digest with the dicts out of an exported
    payload. Normalizing here keeps one implementation of the grouping rather
    than a second copy in the renderer that can drift from this one.
    """

    cluster_id: int
    dominant_diet: str
    other_diet: str
    size: int
    dominant_share: float
    titles: list[str]
    counts: dict[str, int]

    @property
    def dominant_size(self) -> int:
        """How many of the cluster's stories the dominant diet actually ran."""
        return int(self.counts.get(self.dominant_diet, self.size))


def _as_spot(item: object) -> _Spot:
    # Absent-or-None becomes the default; a present value is used as it is. The
    # `x or default` shorthand would be wrong for exactly one field and exactly
    # one value — cluster_id 0, which HDBSCAN hands out on every run — and a
    # cluster whose id reads as -1 silently loses the name it was assigned.
    get = item.get if isinstance(item, dict) else lambda k: getattr(item, k, None)

    def value(key, default):
        found = get(key)
        return default if found is None else found

    return _Spot(
        cluster_id=int(value("cluster_id", -1)),
        dominant_diet=str(value("dominant_diet", "")),
        other_diet=str(value("other_diet", "")),
        size=int(value("size", 0)),
        dominant_share=float(value("dominant_share", 0.0)),
        titles=list(value("representative_titles", [])),
        counts=dict(value("counts", {})),
    )


def group_blindspots(
    blindspots: list,
    assignments: dict[int, ThemeAssignment] | None = None,
    stories_per_theme: int = 8,
) -> list[Theme]:
    """Collapse blindspot clusters into themes, one theme per direction.

    Direction is part of the grouping key, never averaged away. "Modeled CE
    covered this and I didn't" and "I covered this and Modeled CE didn't" are
    different findings about the same subject, and a card that merged them
    would report neither.
    """
    assignments = assignments or {}
    buckets: dict[tuple[str, str], list] = {}
    for raw in blindspots:
        spot = _as_spot(raw)
        assignment = assignments.get(spot.cluster_id) or taxonomy_theme(spot.titles)
        buckets.setdefault((spot.dominant_diet, assignment.key), []).append(
            (assignment, spot)
        )

    themes: list[Theme] = []
    for (dominant, key), members in buckets.items():
        members.sort(key=lambda m: (m[1].size, m[1].dominant_share), reverse=True)
        stories: list[str] = []
        seen: set[str] = set()
        for _, spot in members:
            for title in spot.titles:
                folded = title.casefold()
                if folded not in seen:
                    seen.add(folded)
                    stories.append(title)
        size = sum(spot.size for _, spot in members)
        weighted = sum(spot.size * spot.dominant_share for _, spot in members)
        themes.append(
            Theme(
                key=key,
                title=members[0][0].title,
                dominant_diet=dominant,
                other_diet=members[0][1].other_diet,
                cluster_count=len(members),
                # The dominant diet's own stories, not every member of every
                # cluster: the card's sentence is "this diet ran N stories the
                # other barely touched", and the handful the other diet did run
                # are precisely what makes "barely" the honest word.
                story_count=sum(spot.dominant_size for _, spot in members),
                one_sided=weighted / size if size else 0.0,
                stories=stories[:stories_per_theme],
                method=members[0][0].method,
            )
        )
    # "Other coverage" last within its direction: it is the bucket for what the
    # taxonomy could not name, so it is the least informative card on the page.
    themes.sort(
        key=lambda t: (t.key == OTHER_KEY, -t.story_count, -t.cluster_count, t.title)
    )
    return themes
