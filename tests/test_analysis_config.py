"""Tests for scripts/analysis_config.py.

The real config/ files are read but never written. Every failure case builds a
deliberately malformed YAML file under tmp_path instead, so a broken test can
never corrupt the pre-registered anchor list.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

from analysis_config import (
    Anchor,
    AnchorConfig,
    ConfigError,
    Domain,
    StopwordPolicy,
    config_sha256,
    load_analysis_config,
    load_anchors,
    load_stopword_policy,
    load_yaml,
)

from conftest import base_anchors, base_analysis, base_stopwords  # noqa: E402


# --------------------------------------------------------------------------
# the real anchors.yaml - the pre-registered seed dictionary
# --------------------------------------------------------------------------


def test_real_anchors_has_71_anchors_across_4_domains(real_anchors):
    assert isinstance(real_anchors, AnchorConfig)
    assert len(real_anchors.domains) == 4
    assert len(real_anchors.anchors) == 71


def test_real_anchors_have_no_duplicate_terms(real_anchors):
    terms = [a.term for a in real_anchors.anchors]
    assert len(set(terms)) == len(terms)


def test_real_anchors_have_no_surface_form_collisions(real_anchors):
    forms = [form for a in real_anchors.anchors for form in a.surface_forms]
    assert len(set(forms)) == len(forms)


def test_real_anchors_surface_to_term_covers_every_anchor(real_anchors):
    mapping = real_anchors.surface_to_term
    for anchor in real_anchors.anchors:
        assert mapping[anchor.surface_forms[0]] == anchor.term


def test_real_anchors_multiword_anchors_are_flagged(real_anchors):
    multiword = [a for a in real_anchors.anchors if a.is_multiword]
    assert multiword, "no multi-word anchors found; the collapser would be pointless"
    for anchor in multiword:
        assert max(len(f.split()) for f in anchor.surface_forms) == anchor.ngram


def test_real_anchors_by_term_and_missing_term(real_anchors):
    term = real_anchors.anchors[0].term
    assert real_anchors.by_term(term).term == term
    with pytest.raises(KeyError):
        real_anchors.by_term("definitely_not_an_anchor")


def test_real_anchors_comparisons_reference_defined_domains(real_anchors):
    for comparison in real_anchors.comparisons:
        assert comparison.left in real_anchors.domains
        assert comparison.right in real_anchors.domains


def test_real_anchors_blocklist_is_lowercased(real_anchors):
    assert "reddit" in real_anchors.neighbour_blocklist
    assert all(w == w.lower() for w in real_anchors.neighbour_blocklist)


# --------------------------------------------------------------------------
# anchors - failure cases, all on throwaway YAML
# --------------------------------------------------------------------------


def test_load_anchors_accepts_the_valid_fixture(write_yaml, anchors_data):
    config = load_anchors(write_yaml("anchors.yaml", anchors_data))
    assert set(config.domains) == {"temporal", "absence"}
    assert len(config.anchors) == 3


def test_load_anchors_rejects_duplicate_term_across_domains(write_yaml, anchors_data):
    anchors_data["domains"]["absence"]["anchors"].append(
        {"term": "timezone", "surface_forms": ["tz"]}
    )
    with pytest.raises(ConfigError, match="duplicate term"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_surface_form_collision(write_yaml, anchors_data):
    anchors_data["domains"]["absence"]["anchors"].append(
        {"term": "vanished", "surface_forms": ["weekend"]}
    )
    with pytest.raises(ConfigError, match="already"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_unknown_anchor_key(write_yaml, anchors_data):
    anchors_data["domains"]["temporal"]["anchors"][0]["surfce_forms"] = ["typo"]
    with pytest.raises(ConfigError, match="unknown key"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_empty_domain(write_yaml, anchors_data):
    anchors_data["domains"]["absence"]["anchors"] = []
    with pytest.raises(ConfigError, match="must not be empty"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_comparison_to_undefined_domain(write_yaml, anchors_data):
    anchors_data["comparisons"][0]["right"] = "identity"
    with pytest.raises(ConfigError, match="not a defined domain"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_ngram_wider_than_any_surface_form(write_yaml, anchors_data):
    anchors_data["domains"]["absence"]["anchors"][0]["ngram"] = 3
    with pytest.raises(ConfigError, match="ngram=3"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_unknown_tier(write_yaml, anchors_data):
    anchors_data["domains"]["absence"]["anchors"][0]["tier"] = "medium"
    with pytest.raises(ConfigError, match="not in"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_unknown_match_mode(write_yaml, anchors_data):
    anchors_data["defaults"]["match"] = "fuzzy"
    with pytest.raises(ConfigError, match="defaults.match"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_rejects_missing_domains_key(write_yaml, anchors_data):
    del anchors_data["domains"]
    with pytest.raises(ConfigError, match="missing required key 'domains'"):
        load_anchors(write_yaml("anchors.yaml", anchors_data))


def test_load_anchors_derives_ngram_from_the_widest_surface_form(write_yaml, anchors_data):
    config = load_anchors(write_yaml("anchors.yaml", anchors_data))
    # "time zone" is a spaced variant of a single-word anchor; only a bigram
    # scan can ever match it, so ngram must be 2 regardless of what is declared.
    assert config.by_term("timezone").ngram == 2
    assert config.by_term("timezone").is_multiword is True


def test_load_yaml_reports_a_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_yaml(tmp_path / "absent.yaml")


def test_load_yaml_rejects_a_non_mapping_document(write_text_yaml):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_yaml(write_text_yaml("list.yaml", "- a\n- b\n"))


def test_load_yaml_reports_invalid_yaml(write_text_yaml):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(write_text_yaml("bad.yaml", "a: [1, 2\nb: }{\n"))


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------


def test_anchor_config_domains_cannot_be_assigned_into(write_yaml, anchors_data):
    config = load_anchors(write_yaml("anchors.yaml", anchors_data))
    assert isinstance(config.domains, MappingProxyType)
    with pytest.raises(TypeError):
        config.domains["temporal"] = None


def test_anchor_is_a_frozen_dataclass(write_yaml, anchors_data):
    config = load_anchors(write_yaml("anchors.yaml", anchors_data))
    anchor = config.anchors[0]
    assert dataclasses.is_dataclass(Anchor)
    assert Anchor.__dataclass_params__.frozen is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        anchor.term = "hacked"


def test_domain_and_anchor_collections_are_tuples(write_yaml, anchors_data):
    config = load_anchors(write_yaml("anchors.yaml", anchors_data))
    domain = config.domains["temporal"]
    assert isinstance(domain, Domain)
    assert isinstance(domain.anchors, tuple)
    assert isinstance(domain.terms, tuple)
    assert isinstance(config.anchors[0].surface_forms, tuple)


def test_surface_to_term_mapping_is_read_only(write_yaml, anchors_data):
    config = load_anchors(write_yaml("anchors.yaml", anchors_data))
    with pytest.raises(TypeError):
        config.surface_to_term["weekend"] = "nope"


# --------------------------------------------------------------------------
# stopwords
# --------------------------------------------------------------------------


def test_load_stopword_policy_accepts_the_valid_fixture(write_yaml, stopwords_data):
    policy = load_stopword_policy(write_yaml("stopwords.yaml", stopwords_data))
    assert isinstance(policy, StopwordPolicy)
    assert policy.source == "none"
    assert "reddit" in policy.effective
    assert policy.is_stopword("REDDIT") is True
    assert policy.is_stopword("timezone") is False


def test_load_stopword_policy_rejects_the_yaml_boolean_trap(write_text_yaml):
    # Bare `no` parses as False under YAML 1.1. Silently dropping it from the
    # protect list would let spaCy delete the negator, inverting meaning.
    path = write_text_yaml(
        "stopwords.yaml",
        "version: 1\nsource: none\nprotect:\n  - not\n  - no\nadd: []\nremove: []\n",
    )
    with pytest.raises(ConfigError, match="quote them"):
        load_stopword_policy(path)


def test_load_stopword_policy_rejects_a_non_list_section(write_yaml, stopwords_data):
    stopwords_data["add"] = "reddit"
    with pytest.raises(ConfigError, match="expected a list"):
        load_stopword_policy(write_yaml("stopwords.yaml", stopwords_data))


def test_load_stopword_policy_rejects_unknown_source(write_yaml, stopwords_data):
    stopwords_data["source"] = "nltk"
    with pytest.raises(ConfigError, match="expected 'spacy' or 'none'"):
        load_stopword_policy(write_yaml("stopwords.yaml", stopwords_data))


def test_real_stopword_policy_protects_negators_but_keeps_true_stopwords(real_stopwords):
    assert real_stopwords.source == "spacy"
    for protected in ("not", "no", "never", "alone", "gone", "miss"):
        assert protected not in real_stopwords.effective, (
            f"{protected!r} is on the protect list but survived into the effective stoplist"
        )
    assert "the" in real_stopwords.effective
    assert "reddit" in real_stopwords.effective  # from `add`


def test_real_stopword_policy_effective_list_is_frozen(real_stopwords):
    assert isinstance(real_stopwords.effective, frozenset)
    assert isinstance(real_stopwords.protect, frozenset)


# --------------------------------------------------------------------------
# analysis knobs
# --------------------------------------------------------------------------


def test_load_analysis_config_reads_the_fixture(write_yaml, analysis_data):
    config = load_analysis_config(write_yaml("analysis.yaml", analysis_data))
    assert config.seed == 20260726
    assert config.get("windows.size") == 5
    assert config.get("tiers.tier1.min_freq") == 50


def test_load_analysis_config_seed_override(write_yaml, analysis_data):
    config = load_analysis_config(write_yaml("analysis.yaml", analysis_data), seed=7)
    assert config.seed == 7


def test_load_analysis_config_rejects_inverted_tier_thresholds(write_yaml, analysis_data):
    analysis_data["tiers"]["tier1"]["min_freq"] = 20
    with pytest.raises(ConfigError, match="must exceed"):
        load_analysis_config(write_yaml("analysis.yaml", analysis_data))


def test_load_analysis_config_rejects_equal_tier_thresholds(write_yaml, analysis_data):
    analysis_data["tiers"]["tier1"]["min_freq"] = analysis_data["tiers"]["tier2"]["min_freq"]
    with pytest.raises(ConfigError, match="must exceed"):
        load_analysis_config(write_yaml("analysis.yaml", analysis_data))


def test_load_analysis_config_requires_top_level_sections(write_yaml, analysis_data):
    del analysis_data["figures"]
    with pytest.raises(ConfigError, match="missing required key 'figures'"):
        load_analysis_config(write_yaml("analysis.yaml", analysis_data))


def test_load_analysis_config_requires_path_keys(write_yaml, analysis_data):
    del analysis_data["paths"]["outputs_dir"]
    with pytest.raises(ConfigError, match="outputs_dir"):
        load_analysis_config(write_yaml("analysis.yaml", analysis_data))


def test_analysis_config_get_missing_key_without_default_raises(write_yaml, analysis_data):
    config = load_analysis_config(write_yaml("analysis.yaml", analysis_data))
    with pytest.raises(ConfigError, match="missing key 'windows.nope'"):
        config.get("windows.nope")


def test_analysis_config_get_returns_the_supplied_default(write_yaml, analysis_data):
    config = load_analysis_config(write_yaml("analysis.yaml", analysis_data))
    assert config.get("windows.nope", 3) == 3


def test_analysis_config_section_and_path(write_yaml, analysis_data):
    config = load_analysis_config(write_yaml("analysis.yaml", analysis_data))
    assert config.section("windows")["size"] == 5
    assert config.path("outputs_dir").name == "outputs"
    with pytest.raises(ConfigError, match="expected a mapping"):
        config.section("seed")


def test_analysis_config_raw_is_immutable(write_yaml, analysis_data):
    config = load_analysis_config(write_yaml("analysis.yaml", analysis_data))
    assert isinstance(config.raw, MappingProxyType)
    with pytest.raises(TypeError):
        config.raw["seed"] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1


# --------------------------------------------------------------------------
# config_sha256
# --------------------------------------------------------------------------


def test_config_sha256_is_stable_and_content_sensitive(write_yaml, tmp_path):
    first = write_yaml("a.yaml", {"x": 1})
    assert config_sha256(first) == config_sha256(first)
    first.write_text("x: 2\n", encoding="utf-8")
    assert config_sha256(first) != config_sha256(write_yaml("b.yaml", {"x": 1}))


def test_config_sha256_tolerates_a_missing_file(tmp_path):
    assert len(config_sha256(tmp_path / "gone.yaml")) == 64


def test_config_sha256_is_order_independent(write_yaml):
    a = write_yaml("a.yaml", {"x": 1})
    b = write_yaml("b.yaml", {"y": 2})
    assert config_sha256(a, b) == config_sha256(b, a)


# The module-level helpers exist so the failure fixtures stay honest about what
# a *valid* config looks like; assert they still parse.
def test_fixture_bases_are_valid(write_yaml):
    assert load_anchors(write_yaml("a1.yaml", base_anchors())).anchors
    assert load_analysis_config(write_yaml("a2.yaml", base_analysis())).seed
    assert load_stopword_policy(write_yaml("a3.yaml", base_stopwords())).effective
