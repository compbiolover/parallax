"""Group blindspot clusters into themes a person can scan.

The blindspot engine finds coverage asymmetry at the level of a *cluster*, and
a cluster is small on purpose — HDBSCAN in ``leaf`` mode splits the news finely
so that "one cluster" really is "one story". That is the right unit to measure
and the wrong unit to read: a day produces a couple of dozen of them, each
labeled with whatever terms c-TF-IDF found distinctive, and the reader gets a
list too long to finish titled in a vocabulary nobody speaks.

Three units, and keeping them apart is what this module is for:

* an **article** is one outlet's telling of something;
* a **story** is the articles of one cluster that share a subject, with the
  outlets that carried it — the unit a reader recognizes as "a thing that
  happened";
* a **theme** collects the stories of one subject in one direction, under a
  name written for a human: "Faith & the church", not "christianity today ·
  christianity · today".

The subject is decided **per story, not per cluster**. Clusters are impure
often enough to matter, and labeling the whole cluster propagated its plurality
to every headline inside it: four church headlines and one about a running back
put the running back under "Faith & the church". Per-story assignment splits
that cluster across two themes, which is the honest description of what it was.

Two ways to get the subject, in order of preference:

1. **Claude**, given the headlines and the vocabulary below. Better on subjects
   the taxonomy has never heard of.
2. **The taxonomy** in :data:`TAXONOMY` — a keyword map, versioned in code,
   which always runs and needs no key. It is also the fallback for any single
   story Claude declines to place.

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

class _Unset:
    """Distinguishes "not configured" from a configured ``None``.

    ``cluster.themes.effort`` has three states, and ``None`` cannot carry two of
    them: absent means "use the default below", ``effort: ~`` means "send no
    effort at all and let the model decide", and a string means that string. A
    plain ``None`` default collapsed the first two, so the documented way to
    restore the model's own behaviour silently did nothing.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"


UNSET = _Unset()

# Thinking depth, the same knob the summary and liberty tagging already expose.
# Naming a story's subject from its headline is classification against a fixed
# vocabulary, not reasoning — so `low`, for the reason liberty uses `low`: this
# model runs adaptive thinking at `high` by default, which bills several times
# the output tokens for a judgment that does not improve with them. Raise it in
# settings if theme titles start reading thin.
DEFAULT_THEME_EFFORT = "low"

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
    """Which theme one story belongs to, and who decided."""

    key: str
    title: str
    method: str  # 'taxonomy' | 'claude'


@dataclass(frozen=True)
class Article:
    """One outlet's telling of a story."""

    doc_id: str
    title: str
    url: str | None = None
    outlet: str = ""
    # The registry key, kept as the last resort for naming the outlet: a store
    # ingested before outlet names were recorded has the key and nothing else.
    source_id: str = ""


@dataclass
class Story:
    """One event, and who carried it.

    The unit a reader recognizes. "Iran shutters the country's last
    Presbyterian church, in Christianity Today and two others" is a thing that
    happened; three separate headlines in a list are three sentences they have
    to reassemble into it themselves.
    """

    cluster_id: int
    title: str
    articles: int
    outlets: list[tuple[str, str | None]] = field(default_factory=list)
    one_sided: float = 1.0

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "title": self.title,
            "articles": self.articles,
            "outlets": [{"label": label, "url": url} for label, url in self.outlets],
            "one_sided": self.one_sided,
        }


@dataclass
class Theme:
    """One direction of asymmetry on one subject — the email's card."""

    key: str
    title: str
    dominant_diet: str
    other_diet: str
    stories: list[Story] = field(default_factory=list)
    method: str = "taxonomy"

    @property
    def story_count(self) -> int:
        return len(self.stories)

    @property
    def article_count(self) -> int:
        return sum(s.articles for s in self.stories)

    @property
    def one_sided(self) -> float:
        """Weighted by articles: a ten-article story says more about how
        one-sided this theme is than a one-article story does."""
        total = self.article_count
        if not total:
            return 0.0
        return sum(s.one_sided * s.articles for s in self.stories) / total

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "dominant_diet": self.dominant_diet,
            "other_diet": self.other_diet,
            "story_count": self.story_count,
            "article_count": self.article_count,
            "one_sided": self.one_sided,
            "stories": [s.to_dict() for s in self.stories],
            "method": self.method,
        }


# -- taxonomy ---------------------------------------------------------------


def taxonomy_theme(titles: list[str]) -> ThemeAssignment:
    """Best-matching taxonomy entry for one story's headlines.

    Ranked on *reach* first — how many of the headlines a theme touches at all
    — then on specificity, and only then on raw keyword hits. Hits alone hand
    almost everything to the broadest categories: "Senate Republicans weigh an
    abortion bill" is one abortion word against three of politics', and the
    story is not about the Senate. Reach asks the question that distinguishes
    them, and taxonomy order settles the rest, which is why the catch-alls are
    written last.

    No hits at all is an honest answer, not a failure: those collect under
    "Other coverage" rather than being forced into a category.

    Callers pass **one story's** headlines, not a cluster's. Scoring a whole
    cluster at once is what filed "Cam Skattebo's backflipping at Fanatics Fest"
    under "Faith & the church": the four church headlines beside it won the
    cluster, and the sports headline inherited their label.
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
    entries: list[tuple[str, list[str]]],
    client: object | None = None,
    model: str = DEFAULT_THEME_MODEL,
    effort: str | None = DEFAULT_THEME_EFFORT,
) -> dict[str, ThemeAssignment]:
    """Ask Claude to theme each story. ``{}`` when it cannot or will not.

    Entries are keyed by document id and referenced in the prompt by position:
    the model reads and echoes a small integer rather than a content hash,
    which is both cheaper and one fewer thing for it to get wrong.

    Every failure path returns ``{}`` rather than raising: the taxonomy is
    always there behind this, and a themed brief that is one day less clever
    beats no brief. Anything Claude returns that does not validate — an index
    out of range, a key that is not a key, a title long enough to break the
    card — is dropped for that story alone, which then falls back on its own.
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
        text = _call(client, model, entries, effort)
        raw = _parse(text)
    except Exception as exc:  # network, quota, malformed JSON — all the same here
        logger.warning("Claude theming failed, using the taxonomy: %s", exc)
        return {}

    doc_ids = [doc_id for doc_id, _ in entries]
    out: dict[str, ThemeAssignment] = {}
    for item in raw:
        try:
            index = int(item["story"])
            key = str(item["key"]).strip().lower()
            title = " ".join(str(item["title"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= index < len(doc_ids):
            continue
        doc_id = doc_ids[index]
        if doc_id in out:
            continue
        if not _KEY_OK.match(key) or not _TITLE_OK.match(title):
            continue
        if len(title) > _MAX_TITLE_CHARS:
            continue
        out[doc_id] = ThemeAssignment(key, title, "claude")
    return out


def _call(
    client,
    model: str,
    entries: list[tuple[str, list[str]]],
    effort: str | None = DEFAULT_THEME_EFFORT,
) -> str:
    vocabulary = ", ".join(f"{key} ({title})" for key, title, _ in TAXONOMY)
    lines = [
        "Assign every story below to a theme.",
        "",
        f"Vocabulary of preferred keys: {vocabulary}, {OTHER_KEY} ({OTHER_TITLE}).",
        "",
        'Respond with JSON only: {"assignments": [{"story": 0, "key": "faith", '
        '"title": "Faith & the church"}]}. Every story number below must appear '
        "exactly once. Stories sharing a key must share the same title. Judge "
        "each story on its own headline — neighbouring stories are unrelated.",
        "",
    ]
    for index, (_, titles) in enumerate(entries):
        headline = titles[0] if titles else ""
        lines.append(f"{index}: {headline}")
    kwargs = {
        "model": model,
        # One assignment object runs ~30 tokens now that a story is one line.
        # A fixed ceiling would truncate mid-JSON on a big day and drop the
        # whole batch to the taxonomy — the failure that looks like nothing
        # happening.
        "max_tokens": min(8000, 400 + 40 * len(entries)),
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": "\n".join(lines)}],
    }
    # `max_tokens` caps thinking and the JSON together, so effort is not only a
    # cost lever here: at a high effort on a big day the reasoning can eat the
    # budget and truncate the response, which drops the whole batch to the
    # taxonomy. Omitted entirely when unset, so the model's own default stands.
    if effort:
        kwargs["output_config"] = {"effort": effort}
    resp = client.messages.create(**kwargs)
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
    entries: list[tuple[str, list[str]]],
    client: object | None = None,
    model: str = DEFAULT_THEME_MODEL,
    use_claude: bool = True,
    effort: str | None = DEFAULT_THEME_EFFORT,
) -> dict[str, ThemeAssignment]:
    """Theme every story: Claude where it answered, the taxonomy everywhere else."""
    from_claude = (claude_assignments(entries, client, model, effort)
                   if use_claude else {})
    # A key Claude used keeps Claude's wording everywhere, so two stories it
    # placed together do not arrive under two spellings of one theme.
    titles = {a.key: a.title for a in from_claude.values()}
    out: dict[str, ThemeAssignment] = {}
    for doc_id, story_titles in entries:
        assignment = from_claude.get(doc_id) or taxonomy_theme(story_titles)
        title = titles.get(assignment.key, assignment.title)
        out[doc_id] = ThemeAssignment(assignment.key, title, assignment.method)
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
    assignments: dict[str, ThemeAssignment] | None = None,
    articles: dict[int, list[Article]] | None = None,
    stories_per_theme: int = 6,
) -> list[Theme]:
    """Group a run's blindspots into themes, by story rather than by cluster.

    Two levels, and the distinction is the point of this function:

    * a **story** is one event — the articles of one cluster that share a
      subject, with the outlets that carried it;
    * a **theme** collects the stories of one subject in one direction.

    The theme is decided per story, not per cluster. Clusters are impure often
    enough to matter: a cluster of four church headlines and one about a
    running back is one cluster, and labeling the cluster put the running back
    under "Faith & the church". Assigning per story splits that cluster across
    two themes, which is the honest description of what it was.

    Direction stays part of the grouping key, never averaged away. "They
    covered this and I didn't" and the reverse are different findings about the
    same subject, and a card merging them would report neither.
    """
    assignments = assignments or {}
    articles = articles or {}
    # (dominant_diet, theme_key) -> cluster_id -> [(assignment, article, spot)]
    buckets: dict[tuple[str, str], dict[int, list]] = {}
    for raw in blindspots:
        spot = _as_spot(raw)
        for article in articles.get(spot.cluster_id) or _articles_from_titles(spot):
            assignment = (assignments.get(article.doc_id)
                          or taxonomy_theme([article.title]))
            by_cluster = buckets.setdefault((spot.dominant_diet, assignment.key), {})
            by_cluster.setdefault(spot.cluster_id, []).append((assignment, article, spot))

    themes: list[Theme] = []
    for (dominant, key), by_cluster in buckets.items():
        stories = [_story(cluster_id, rows) for cluster_id, rows in by_cluster.items()]
        stories.sort(key=lambda s: (s.articles, s.title), reverse=True)
        members = [row for rows in by_cluster.values() for row in rows]
        first = members[0]
        # The name, and who wrote it. `assign_themes` already unifies the title
        # across a key, so one Claude assignment anywhere in the bucket means
        # the words on the card are Claude's whatever placed the other stories
        # — and a story that fell back to the taxonomy in here has not been
        # through that unification. Taking the first story's title and method
        # therefore both mis-worded the card and credited the wrong author.
        named = next((a for a, _, _ in members if a.method == "claude"), first[0])
        themes.append(
            Theme(
                key=key,
                title=named.title,
                dominant_diet=dominant,
                other_diet=first[2].other_diet,
                stories=stories[:stories_per_theme],
                method=named.method,
            )
        )
    # "Other coverage" last within its direction: it is the bucket for what
    # nothing could name, so it is the least informative card on the page.
    themes.sort(
        key=lambda t: (t.key == OTHER_KEY, -t.article_count, -t.story_count, t.title)
    )
    return themes


def _story(cluster_id: int, rows: list) -> Story:
    """One cluster's articles on one subject, as a story.

    The lead headline is the longest, on the same reasoning the representative
    titles use: length carries story detail once the outlet stamp is off. The
    outlets are de-duplicated by name, because two articles from one masthead
    are one outlet covering a story twice, not two outlets covering it.
    """
    titles = sorted((a.title for _, a, _ in rows), key=len, reverse=True)
    outlets: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for _, article, _ in rows:
        label = (article.outlet or _host(article.url)
                 or _from_source_id(article.source_id))
        if not label or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        outlets.append((label, article.url))
    spot = rows[0][2]
    return Story(
        cluster_id=cluster_id,
        title=titles[0] if titles else "(untitled story)",
        articles=len(rows),
        outlets=outlets,
        one_sided=spot.dominant_share,
    )


def _articles_from_titles(spot: _Spot) -> list[Article]:
    """What a payload exported before stories existed can still offer.

    Headlines with no outlet and no link. The grouping is the same; the card
    just has less to show.
    """
    return [Article(doc_id=f"{spot.cluster_id}:{i}", title=title)
            for i, title in enumerate(spot.titles)]


def _from_source_id(source_id: str) -> str:
    """``christianity_today`` -> "Christianity Today". A last resort.

    Outlet names are recorded at ingestion, so this only fires for a store
    ingested before that was true — but dropping the outlet entirely there
    costs the reader the one thing that makes a story checkable, and a
    de-slugged key is a recognizable masthead far more often than not.
    Initialisms stay upper-case, since the ids use them (``npr`` -> "NPR").
    """
    words = [w for w in re.split(r"[_\-\s]+", source_id or "") if w]
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in words)


def _host(url: str | None) -> str:
    """A bare hostname, for an article whose outlet has no recorded name."""
    if not url:
        return ""
    host = re.sub(r"^\w+://", "", url).split("/")[0].split("@")[-1]
    return host[4:] if host.startswith("www.") else host
