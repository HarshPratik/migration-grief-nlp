"""Phases 2-4 - load and validate the analysis configuration.

Shared module, imported by scripts 07-21 the same way `seen_ids` and
`scrape_plan` are imported by scripts 01-06:

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "scripts"))
    from analysis_config import load_all

Everything returned is immutable (frozen dataclasses, tuples, MappingProxyType)
so a downstream script cannot mutate config and change another script's
behaviour. Validation is strict and fails loudly: a typo in a config key is a
non-zero exit, never a silently-ignored setting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"
ANALYSIS_PATH = CONFIG_DIR / "analysis.yaml"
ANCHORS_PATH = CONFIG_DIR / "anchors.yaml"
STOPWORDS_PATH = CONFIG_DIR / "stopwords.yaml"

VALID_TIERS = ("high", "mid", "low", "rare")
VALID_MATCH = ("surface", "lemma", "regex")
ANCHOR_KEYS = {"term", "surface_forms", "variants", "ngram", "tier", "pos", "notes", "polysemous"}
DOMAIN_KEYS = {"label", "subtitle", "color", "description", "anchors"}


class ConfigError(Exception):
    """Raised when a config file is malformed. Message always names the file and key path."""


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    """Read a YAML mapping, with errors that name the file."""
    if not path.exists():
        raise ConfigError(f"{path}: config file not found")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML - {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def config_sha256(*paths: Path) -> str:
    """Stable hash of the config files that produced a result.

    Embedded in every summary JSON so a reviewer asking "which anchor list
    produced Table 3" gets an exact answer.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively convert to immutable containers."""
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _require(data: Mapping, key: str, where: str, kind: type | tuple[type, ...] | None = None) -> Any:
    if key not in data:
        raise ConfigError(f"{where}: missing required key '{key}'")
    value = data[key]
    if kind is not None and not isinstance(value, kind):
        names = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise ConfigError(f"{where}.{key}: expected {names}, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Anchor:
    """One seed-dictionary entry, as written by the lead author."""

    term: str
    domain: str
    surface_forms: tuple[str, ...]
    ngram: int = 1
    tier: str = "rare"
    pos: str | None = None
    notes: str | None = None
    polysemous: bool = False

    @property
    def is_multiword(self) -> bool:
        return self.ngram > 1


@dataclass(frozen=True, slots=True)
class Domain:
    """One of the four theoretical concept domains."""

    name: str
    label: str
    description: str
    color: str
    anchors: tuple[Anchor, ...]
    subtitle: str = ""

    @property
    def terms(self) -> tuple[str, ...]:
        return tuple(a.term for a in self.anchors)


@dataclass(frozen=True, slots=True)
class Comparison:
    """A declared cross-domain hypothesis (one test in family F1)."""

    name: str
    left: str
    right: str
    hypothesis: str


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    domains: Mapping[str, Domain]
    comparisons: tuple[Comparison, ...]
    neighbour_blocklist: frozenset[str]
    defaults: Mapping[str, Any]

    @property
    def anchors(self) -> tuple[Anchor, ...]:
        return tuple(a for d in self.domains.values() for a in d.anchors)

    def by_term(self, term: str) -> Anchor:
        for anchor in self.anchors:
            if anchor.term == term:
                return anchor
        raise KeyError(f"no anchor with term {term!r}")

    @property
    def surface_to_term(self) -> Mapping[str, str]:
        """Every matchable surface form -> its canonical term.

        Consumers should iterate this longest-first so that `video call` wins
        over `call`; see text_windows.collapse_phrases.
        """
        return MappingProxyType(
            {sf: a.term for a in self.anchors for sf in a.surface_forms}
        )


def _parse_anchor(raw: Any, domain: str, index: int, defaults: Mapping) -> Anchor:
    where = f"{ANCHORS_PATH.name}:domains.{domain}.anchors[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    unknown = set(raw) - ANCHOR_KEYS
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. Allowed: {sorted(ANCHOR_KEYS)}"
        )

    term = _require(raw, "term", where, str).strip()
    if not term:
        raise ConfigError(f"{where}.term: must not be empty")

    forms = raw.get("surface_forms") or raw.get("variants") or [term]
    if not isinstance(forms, list) or not all(isinstance(f, str) for f in forms):
        raise ConfigError(f"{where}.surface_forms: expected a list of strings")
    forms = tuple(dict.fromkeys(f.strip().lower() for f in forms if f.strip()))
    if not forms:
        raise ConfigError(f"{where}.surface_forms: resolved to empty")

    tier = raw.get("tier", defaults.get("tier", "rare"))
    if tier not in VALID_TIERS:
        raise ConfigError(f"{where}.tier: {tier!r} not in {VALID_TIERS}")

    # `ngram` is the widest span the collapser must scan for this anchor. It is
    # DERIVED from the surface forms rather than trusted, because a single-word
    # anchor can still carry a spaced spelling variant ("timezone" / "time zone",
    # "wifi" / "wi fi") that only a bigram scan will ever match.
    observed = max(len(f.split()) for f in forms)
    declared = int(raw.get("ngram", observed))
    if declared > observed:
        raise ConfigError(
            f"{where}: ngram={declared} but the longest surface form is {observed} word(s). "
            "Declaring a span no surface form can fill is a typo."
        )

    return Anchor(
        term=term,
        domain=domain,
        surface_forms=forms,
        ngram=observed,
        tier=tier,
        pos=raw.get("pos"),
        notes=raw.get("notes"),
        polysemous=bool(raw.get("polysemous", False)),
    )


def load_anchors(path: Path = ANCHORS_PATH) -> AnchorConfig:
    """Load and validate the seed dictionaries.

    Rejects duplicate terms across domains (a duplicate would silently
    double-count in every frequency table) and surface-form collisions (which
    would make phrase collapsing ambiguous).
    """
    data = load_yaml(path)
    defaults = data.get("defaults") or {}
    match_mode = defaults.get("match", "surface")
    if match_mode not in VALID_MATCH:
        raise ConfigError(f"{path.name}:defaults.match: {match_mode!r} not in {VALID_MATCH}")

    raw_domains = _require(data, "domains", path.name, dict)
    if not raw_domains:
        raise ConfigError(f"{path.name}:domains: must not be empty")

    domains: dict[str, Domain] = {}
    seen_terms: dict[str, str] = {}
    seen_forms: dict[str, str] = {}

    for name, raw in raw_domains.items():
        where = f"{path.name}:domains.{name}"
        if not isinstance(raw, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
        unknown = set(raw) - DOMAIN_KEYS
        if unknown:
            raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")

        raw_anchors = _require(raw, "anchors", where, list)
        if not raw_anchors:
            raise ConfigError(f"{where}.anchors: must not be empty")

        anchors = tuple(
            _parse_anchor(item, name, i, defaults) for i, item in enumerate(raw_anchors)
        )

        for anchor in anchors:
            if anchor.term in seen_terms:
                raise ConfigError(
                    f"{where}: duplicate term {anchor.term!r} - already defined in domain "
                    f"'{seen_terms[anchor.term]}'. A duplicate double-counts in every table."
                )
            seen_terms[anchor.term] = name
            for form in anchor.surface_forms:
                if form in seen_forms:
                    raise ConfigError(
                        f"{where}: surface form {form!r} on anchor {anchor.term!r} already "
                        f"claimed by {seen_forms[form]!r}. Collisions make collapsing ambiguous."
                    )
                seen_forms[form] = anchor.term

        domains[name] = Domain(
            name=name,
            label=_require(raw, "label", where, str),
            description=str(raw.get("description", "")).strip(),
            color=raw.get("color", "#666666"),
            subtitle=str(raw.get("subtitle", "")).strip(),
            anchors=anchors,
        )

    comparisons = []
    for i, raw in enumerate(data.get("comparisons") or []):
        where = f"{path.name}:comparisons[{i}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{where}: expected a mapping")
        left = _require(raw, "left", where, str)
        right = _require(raw, "right", where, str)
        for side, value in (("left", left), ("right", right)):
            if value not in domains:
                raise ConfigError(
                    f"{where}.{side}: {value!r} is not a defined domain. "
                    f"Known: {sorted(domains)}"
                )
        comparisons.append(
            Comparison(
                name=_require(raw, "name", where, str),
                left=left,
                right=right,
                hypothesis=str(raw.get("hypothesis", "")).strip(),
            )
        )

    return AnchorConfig(
        domains=MappingProxyType(domains),
        comparisons=tuple(comparisons),
        neighbour_blocklist=frozenset(
            str(w).lower() for w in (data.get("neighbour_blocklist") or [])
        ),
        defaults=_freeze(defaults),
    )


# --------------------------------------------------------------------------
# stopwords
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StopwordPolicy:
    """Explicit, version-controlled stoplist replacing spaCy's `tok.is_stop`."""

    source: str
    protect: frozenset[str]
    add: frozenset[str]
    remove: frozenset[str]
    _effective: frozenset[str] = field(default=frozenset(), repr=False)

    @property
    def effective(self) -> frozenset[str]:
        return self._effective

    def is_stopword(self, token: str) -> bool:
        return token.lower() in self._effective


def load_stopword_policy(path: Path = STOPWORDS_PATH) -> StopwordPolicy:
    """Build the effective stoplist = source - protect - remove + add.

    Guards against the YAML 1.1 booleans trap: bare `no`, `yes`, `on` and `off`
    parse as booleans, and an unquoted `no` would silently drop a negator from
    the protect list - which inverts sentence meaning downstream.
    """
    data = load_yaml(path)
    source = str(data.get("source", "spacy")).lower()
    if source not in ("spacy", "none"):
        raise ConfigError(f"{path.name}:source: expected 'spacy' or 'none', got {source!r}")

    lists: dict[str, frozenset[str]] = {}
    for key in ("protect", "add", "remove"):
        values = data.get(key) or []
        if not isinstance(values, list):
            raise ConfigError(f"{path.name}:{key}: expected a list")
        bad = [v for v in values if not isinstance(v, str)]
        if bad:
            raise ConfigError(
                f"{path.name}:{key}: non-string entr{'y' if len(bad) == 1 else 'ies'} {bad!r}. "
                "YAML parses bare no/yes/on/off as booleans - quote them."
            )
        lists[key] = frozenset(v.strip().lower() for v in values if v.strip())

    if source == "spacy":
        from spacy.lang.en.stop_words import STOP_WORDS

        base = frozenset(w.lower() for w in STOP_WORDS)
    else:
        base = frozenset()

    effective = (base - lists["protect"] - lists["remove"]) | lists["add"]

    return StopwordPolicy(
        source=source,
        protect=lists["protect"],
        add=lists["add"],
        remove=lists["remove"],
        _effective=frozenset(effective),
    )


# --------------------------------------------------------------------------
# analysis knobs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Knobs from config/analysis.yaml. Access nested values with `get`."""

    seed: int
    raw: Mapping[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value: cfg.get('windows.size'). Raises if no default given."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is not None:
                    return default
                raise ConfigError(
                    f"{ANALYSIS_PATH.name}: missing key '{dotted}' (no default supplied)"
                )
            node = node[part]
        return node

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{ANALYSIS_PATH.name}:{name}: expected a mapping")
        return value

    def path(self, key: str) -> Path:
        """Resolve a configured directory to an absolute path under ROOT."""
        return ROOT / str(self.get(f"paths.{key}"))


def load_analysis_config(path: Path = ANALYSIS_PATH, seed: int | None = None) -> AnalysisConfig:
    data = load_yaml(path)
    for required in ("paths", "tiers", "windows", "figures"):
        _require(data, required, path.name, dict)
    for key in ("input_dir", "analysis_dir", "outputs_dir"):
        _require(data["paths"], key, f"{path.name}:paths", str)

    resolved_seed = int(seed if seed is not None else data.get("seed", 0))

    tiers = data["tiers"]
    t1, t2 = tiers.get("tier1", {}), tiers.get("tier2", {})
    if t1.get("min_freq", 0) <= t2.get("min_freq", 0):
        raise ConfigError(
            f"{path.name}:tiers: tier1.min_freq ({t1.get('min_freq')}) must exceed "
            f"tier2.min_freq ({t2.get('min_freq')})"
        )

    return AnalysisConfig(seed=resolved_seed, raw=_freeze(data))


def load_all(seed: int | None = None) -> tuple[AnalysisConfig, AnchorConfig, StopwordPolicy]:
    """Load all three configs. The single entry point scripts 07-21 should use."""
    return (
        load_analysis_config(seed=seed),
        load_anchors(),
        load_stopword_policy(),
    )
