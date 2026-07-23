"""A small built-in moral-foundations lexicon so the pipeline runs out of the box.

THIS IS A DEMO SEED, NOT A VALIDATED INSTRUMENT. It exists so tests and a first
end-to-end run work with zero external data. For real scoring, point the scorer
at the full eMFD (see ``scoring/lexicon.py`` and ``LIMITATIONS.md``): the eMFD
assigns continuous per-word foundation probabilities over a ~10k-word
vocabulary, whereas this seed is a few dozen hand-picked stems with unit weights.

Entry format: ``stem -> (foundations, pole)`` where ``foundations`` maps a
CLASSIC foundation to a weight in (0, 1], and ``pole`` is +1 for a virtue term
or -1 for a vice term (used for the sentiment signal). A trailing ``*`` marks a
prefix/wildcard stem (MFD-style): ``care*`` matches "care", "caring", "careful".
"""

from __future__ import annotations

# fmt: off
SEED_LEXICON: dict[str, tuple[dict[str, float], int]] = {
    # ---- care / harm ----
    "care":        ({"care": 1.0}, +1), "caring":     ({"care": 1.0}, +1),
    "protect":     ({"care": 1.0}, +1), "compassion": ({"care": 1.0}, +1),
    "shelter":     ({"care": 1.0}, +1), "safe":       ({"care": 1.0}, +1),
    "nurtur":      ({"care": 1.0}, +1), "help":       ({"care": 1.0}, +1),
    "harm":        ({"care": 1.0}, -1), "hurt":       ({"care": 1.0}, -1),
    "suffer":      ({"care": 1.0}, -1), "cruel":      ({"care": 1.0}, -1),
    "violence":    ({"care": 1.0}, -1), "abuse":      ({"care": 1.0}, -1),
    "kill":        ({"care": 1.0}, -1), "victim":     ({"care": 1.0}, -1),

    # ---- fairness / cheating ----
    "fair":        ({"fairness": 1.0}, +1), "equal":     ({"fairness": 1.0}, +1),
    "justice":     ({"fairness": 1.0}, +1), "rights":    ({"fairness": 1.0}, +1),
    "equit":       ({"fairness": 1.0}, +1), "reciprocat":({"fairness": 1.0}, +1),
    "impartial":   ({"fairness": 1.0}, +1),
    "cheat":       ({"fairness": 1.0}, -1), "fraud":     ({"fairness": 1.0}, -1),
    "unfair":      ({"fairness": 1.0}, -1), "biased":    ({"fairness": 1.0}, -1),
    "corrupt":     ({"fairness": 1.0}, -1), "injustice": ({"fairness": 1.0}, -1),
    "discriminat": ({"fairness": 1.0}, -1),

    # ---- loyalty / betrayal ----
    "loyal":       ({"loyalty": 1.0}, +1), "patriot":   ({"loyalty": 1.0}, +1),
    "unite":       ({"loyalty": 1.0}, +1), "solidarity":({"loyalty": 1.0}, +1),
    "community":   ({"loyalty": 1.0}, +1), "nation":    ({"loyalty": 1.0}, +1),
    "together":    ({"loyalty": 1.0}, +1), "family":    ({"loyalty": 1.0}, +1),
    "betray":      ({"loyalty": 1.0}, -1), "traitor":   ({"loyalty": 1.0}, -1),
    "disloyal":    ({"loyalty": 1.0}, -1), "treason":   ({"loyalty": 1.0}, -1),
    "abandon":     ({"loyalty": 1.0}, -1),

    # ---- authority / subversion ----
    "authority":   ({"authority": 1.0}, +1), "obey":     ({"authority": 1.0}, +1),
    "tradition":   ({"authority": 1.0}, +1), "order":    ({"authority": 1.0}, +1),
    "respect":     ({"authority": 1.0}, +1), "duty":     ({"authority": 1.0}, +1),
    "law":         ({"authority": 1.0}, +1), "leader":   ({"authority": 1.0}, +1),
    "rebel":       ({"authority": 1.0}, -1), "defy":     ({"authority": 1.0}, -1),
    "disobey":     ({"authority": 1.0}, -1), "subvert":  ({"authority": 1.0}, -1),
    "riot":        ({"authority": 1.0}, -1), "chaos":    ({"authority": 1.0}, -1),

    # ---- sanctity / degradation ----
    "sacred":      ({"sanctity": 1.0}, +1), "holy":      ({"sanctity": 1.0}, +1),
    "pure":        ({"sanctity": 1.0}, +1), "faith":     ({"sanctity": 1.0}, +1),
    "sanctit":     ({"sanctity": 1.0}, +1), "bless":     ({"sanctity": 1.0}, +1),
    "dignity":     ({"sanctity": 1.0}, +1), "worship":   ({"sanctity": 1.0}, +1),
    "sin":         ({"sanctity": 1.0}, -1), "impure":    ({"sanctity": 1.0}, -1),
    "degrad":      ({"sanctity": 1.0}, -1), "obscene":   ({"sanctity": 1.0}, -1),
    "defil":       ({"sanctity": 1.0}, -1), "corruption":({"sanctity": 1.0}, -1),
}
# fmt: on
