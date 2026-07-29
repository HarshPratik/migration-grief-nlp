"""Phases 2-4 - phrase collapsing, token indexing and context-window extraction.

Shared module for scripts 07-21. Every function here is pure: inputs are never
mutated, outputs are always new objects.

This is where the +/-5 word windows behind Phase 3 come from, so the boundary
rules are spelled out explicitly rather than left implicit:

* windows are clipped at sentence boundaries - a window spilling into the next
  sentence measures that sentence's emotion, not the anchor's;
* clipping can amputate a negator ("I do not | miss home"), which would invert
  the measured emotion, so a negator within `negation_rescue` tokens to the left
  is pulled back in;
* multi-word anchors are collapsed to a single token BEFORE any counting, or
  `video call` and `culture shock` can never clear a frequency floor.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_io import as_token_list  # noqa: E402,F401  (re-exported for callers)

# Tokens that flip polarity. Kept here rather than in config because window
# geometry is code, not a tunable: changing this changes what a window IS.
NEGATORS = frozenset({
    "not", "n't", "nt", "no", "never", "none", "nothing", "nobody", "nowhere",
    "neither", "nor", "cannot", "cant", "without", "hardly", "barely", "scarcely",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "wont",
    "couldnt", "shouldnt", "wouldnt", "havent", "hasnt", "hadnt",
})

_WORD_RE = re.compile(r"[a-z0-9']+")


# --------------------------------------------------------------------------
# tokenisation helpers
# --------------------------------------------------------------------------


def simple_tokens(text: str) -> list[str]:
    """Lowercase word tokens. Used for surface matching, not for the NLP pipeline."""
    return _WORD_RE.findall((text or "").lower())


def collapse_phrases(
    tokens: Sequence[str],
    surface_to_term: Mapping[str, str],
    *,
    max_span: int | None = None,
) -> list[str]:
    """Replace multi-word surface forms with their canonical single token.

    Longest match wins, so `video call` beats `call` and `culture shock` beats a
    bare `culture`. Scans left to right and never re-examines consumed tokens.

    >>> collapse_phrases(["i", "had", "a", "video", "call"], {"video call": "video_call"})
    ['i', 'had', 'a', 'video_call']
    """
    if not tokens:
        return []
    if not surface_to_term:
        return [str(t) for t in tokens]

    spans = {len(form.split()) for form in surface_to_term}
    widest = max_span or max(spans)

    lowered = [str(t).lower() for t in tokens]
    out: list[str] = []
    i = 0
    n = len(lowered)
    while i < n:
        matched = False
        # Longest span first - this is what makes longest-match-wins hold.
        for span in range(min(widest, n - i), 0, -1):
            candidate = " ".join(lowered[i : i + span])
            term = surface_to_term.get(candidate)
            if term is not None:
                out.append(term)
                i += span
                matched = True
                break
        if not matched:
            out.append(lowered[i])
            i += 1
    return out


@dataclass(frozen=True, slots=True)
class SurfaceMatcher:
    """Matches anchor surface forms against a real tokenizer's output.

    `collapse_phrases` assumes tokens split on whitespace. A real tokenizer does
    not: spaCy renders "wi-fi" as ["wi", "-", "fi"] and "video call" as
    ["video", "call"], so naive whitespace joining misses both.

    This matcher compares alphanumeric-only concatenations, so ["wi","-","fi"]
    and "wi-fi" both reduce to "wifi".

    The single/multi split matters for precision: only surface forms that are
    genuinely multi-word may match across token boundaries. Without that rule,
    the anchor "away" would swallow "in a way" (a + way -> "away").
    """

    single: Mapping[str, str]      # exact one-token surface -> canonical term
    multi: Mapping[str, str]       # alnum concatenation -> canonical term
    max_span: int

    @staticmethod
    def _key(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @classmethod
    def from_surface_map(cls, surface_to_term: Mapping[str, str], *, max_span: int = 5) -> "SurfaceMatcher":
        single: dict[str, str] = {}
        multi: dict[str, str] = {}
        for surface, term in surface_to_term.items():
            lowered = surface.lower().strip()
            if " " in lowered or "-" in lowered:
                key = cls._key(lowered)
                if key:
                    multi[key] = term
            else:
                single[lowered] = term
        return cls(single=single, multi=multi, max_span=max_span)

    def collapse(self, tokens: Sequence[str]) -> list[tuple[str, int, int]]:
        """Return [(emitted_token, start_index, span_length)] over the input.

        Longest span wins. Non-anchor tokens pass through unchanged.
        """
        out: list[tuple[str, int, int]] = []
        i = 0
        n = len(tokens)
        while i < n:
            matched = False
            for span in range(min(self.max_span, n - i), 1, -1):
                key = self._key("".join(str(t) for t in tokens[i : i + span]))
                if key and key in self.multi:
                    out.append((self.multi[key], i, span))
                    i += span
                    matched = True
                    break
            if matched:
                continue
            lowered = str(tokens[i]).lower()
            term = self.single.get(lowered)
            out.append((term if term is not None else lowered, i, 1))
            i += 1
        return out


def build_token_index(tokens: Sequence[str]) -> dict[str, list[int]]:
    """token -> ascending positions.

    Built once per document so window extraction is O(hits) instead of
    O(anchors x tokens); the naive form is ~10^8 comparisons on this corpus.
    """
    index: dict[str, list[int]] = defaultdict(list)
    for position, token in enumerate(tokens):
        index[str(token)].append(position)
    return dict(index)


def vocab_counts(documents: Iterable[Sequence[str]]) -> Counter:
    """Token frequency across documents."""
    counter: Counter = Counter()
    for tokens in documents:
        counter.update(str(t) for t in tokens)
    return counter


def doc_freq(documents: Iterable[Sequence[str]]) -> Counter:
    """Number of documents containing each token (not total occurrences)."""
    counter: Counter = Counter()
    for tokens in documents:
        counter.update(set(str(t) for t in tokens))
    return counter


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Window:
    """One anchor occurrence and the span of context kept around it."""

    anchor: str
    position: int
    start: int
    end: int              # inclusive
    sent_idx: int | None
    truncated_left: bool
    truncated_right: bool
    negation_rescued: bool

    @property
    def n_left(self) -> int:
        return self.position - self.start

    @property
    def n_right(self) -> int:
        return self.end - self.position

    def tokens(self, source: Sequence[str]) -> list[str]:
        return [str(t) for t in source[self.start : self.end + 1]]


def extract_windows(
    tokens: Sequence[str],
    anchor_positions: Mapping[str, Sequence[int]],
    *,
    size: int = 5,
    sent_idx: Sequence[int] | None = None,
    clip_at_sentence: bool = True,
    negation_rescue: int = 0,
    negators: frozenset[str] = NEGATORS,
    max_per_anchor: int | None = None,
) -> list[Window]:
    """Extract +/-`size` token windows around each anchor occurrence.

    Boundary rules, in order of application:
      1. clamp to the document;
      2. if `clip_at_sentence` and `sent_idx` is given, clamp to the anchor's
         own sentence;
      3. if `negation_rescue` > 0, extend the left edge back to the earliest
         negator lying within that many tokens of the current left edge -
         still respecting the sentence boundary.

    Returns windows ordered by (anchor, position). Never mutates its inputs.
    """
    if size < 0:
        raise ValueError(f"window size must be >= 0, got {size}")
    n = len(tokens)
    if n == 0:
        return []
    if sent_idx is not None and len(sent_idx) != n:
        raise ValueError(
            f"sent_idx length {len(sent_idx)} does not match tokens length {n}"
        )

    use_sentences = clip_at_sentence and sent_idx is not None
    windows: list[Window] = []

    for anchor in sorted(anchor_positions):
        positions = sorted(p for p in anchor_positions[anchor] if 0 <= p < n)
        if max_per_anchor is not None:
            positions = positions[:max_per_anchor]

        for position in positions:
            desired_start = position - size
            desired_end = position + size
            start = max(0, desired_start)
            end = min(n - 1, desired_end)

            if use_sentences:
                own = sent_idx[position]
                while start > 0 and sent_idx[start - 1] == own and start > desired_start:
                    start -= 1
                while start < position and sent_idx[start] != own:
                    start += 1
                while end < n - 1 and sent_idx[end + 1] == own and end < desired_end:
                    end += 1
                while end > position and sent_idx[end] != own:
                    end -= 1

            rescued = False
            if negation_rescue > 0 and start > 0:
                floor = max(0, start - negation_rescue)
                for probe in range(floor, start):
                    if use_sentences and sent_idx[probe] != sent_idx[position]:
                        continue
                    if str(tokens[probe]).lower() in negators:
                        start = probe
                        rescued = True
                        break

            windows.append(
                Window(
                    anchor=anchor,
                    position=position,
                    start=start,
                    end=end,
                    sent_idx=int(sent_idx[position]) if sent_idx is not None else None,
                    truncated_left=start > desired_start and not rescued,
                    truncated_right=end < desired_end,
                    negation_rescued=rescued,
                )
            )

    return windows


def dedupe_overlapping(windows: Sequence[Window]) -> list[Window]:
    """Drop later windows of the same anchor whose span is already covered.

    Two occurrences of one anchor within `size` tokens produce near-identical
    context; counting both double-weights that passage.
    """
    kept: list[Window] = []
    last_end: dict[str, int] = {}
    for window in sorted(windows, key=lambda w: (w.anchor, w.position)):
        previous = last_end.get(window.anchor)
        if previous is not None and window.start <= previous:
            continue
        kept.append(window)
        last_end[window.anchor] = window.end
    return kept


def kwic_line(
    tokens: Sequence[str],
    window: Window,
    *,
    separator: str = " | ",
) -> str:
    """Keyword-in-context line: left context | anchor | right context."""
    left = " ".join(str(t) for t in tokens[window.start : window.position])
    node = str(tokens[window.position])
    right = " ".join(str(t) for t in tokens[window.position + 1 : window.end + 1])
    return f"{left}{separator}{node}{separator}{right}"


def find_anchor_positions(
    tokens: Sequence[str],
    terms: Iterable[str],
    *,
    index: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, list[int]]:
    """Positions of each requested term. Terms absent from the document are omitted.

    Assumes `tokens` has already been through `collapse_phrases`, so a
    multi-word anchor is a single token by now.
    """
    token_index = index if index is not None else build_token_index(tokens)
    found: dict[str, list[int]] = {}
    for term in terms:
        positions = token_index.get(term)
        if positions:
            found[term] = list(positions)
    return found
