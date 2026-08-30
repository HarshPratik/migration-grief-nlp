# Migration and Grief Narrative Analysis

Comparative NLP study of two Reddit **post** corpora (no comments):

| Corpus | Theme | Subreddits (10k sample) |
|--------|--------|-------------------------|
| **A — migration** | Expat / immigration / study-abroad experience | `r/expats` 3,889 · `r/StudyAbroad` 3,240 · `r/immigration` 2,871 |
| **B — grief** | Loss and bereavement | `r/GriefSupport` 9,752 · `r/bereavement` 248 |

**Question.** Do migration narratives and grief narratives organise the same four seed domains — **Temporal**, **Identity**, **Absence**, **Digital** — in the same way? We compare collocates (PPMI), frozen BERT senses, trained Word2Vec/FastText, emotion around anchors, co-occurrence networks, and topic models.

**Sample.** 10,000 posts per corpus. Body > 100 words. Window `2023-07-01 ≤ created_at < 2026-08-24`. Sampling seed `20260827`. Collection is **Arctic Shift** (this rebuild does not use Apify). Corpus A drops visa/tax/paperwork Q&A (whole-word filters + procedural flair). Corpus B is scarce-first so `r/bereavement` is fully included.

This README is the results front page. Longer methods notes: [`docs/METHODS_RESULTS.md`](docs/METHODS_RESULTS.md). Sampling audit: [`data/processed/PHASE1_COMPLETE.md`](data/processed/PHASE1_COMPLETE.md).

---

## Contents

1. [Status](#status)
2. [What we measured](#what-we-measured)
3. [Sample and seed dictionary](#sample-and-seed-dictionary)
4. [Phase 2 - semantics](#phase-2---semantics)
5. [Declared pair hypotheses](#declared-pair-hypotheses)
6. [Phase 3 - emotion and co-occurrence](#phase-3---emotion-and-co-occurrence)
7. [Phase 4 - topics](#phase-4---topics)
8. [What holds vs what is weak](#what-holds-vs-what-is-weak)
9. [Figure index](#figure-index)
10. [Reproduce](#reproduce)
11. [Ethics](#ethics)

---

## Status

| | |
|--|--|
| **Computational pipeline** | **Done** on the 10k / 10k analysis frame (Phases 1–4) |
| **Ready for** | Drafting the paper from the results below |
| **Not finished** | Human coding of 150 emotion windows; the paper text itself |
| **Repo** | **Private.** Row-level CSVs contain Reddit post text — do not make public without ethics review |

**Done**

- Phase 1 sample: 10,000 + 10,000 posts; unique IDs; A ∩ B = 0; no empty/`[removed]` bodies.
- Seed dictionary: **4 domains, 71 anchors** (`config/anchors.yaml`).
- Phase 2: PPMI, frozen BERT, Word2Vec, FastText, reliability gates, top-15 neighbours, PCA/t-SNE, methods scoreboard.
- Phase 3: DistilRoBERTa-emotion on ±5-token windows (ABSA **proxy**). All four domains differ after FDR (q = 0.0003).
- Phase 4: pooled NMF **k = 12** (11/12 core topics) **and** independent LDA/NMF (k = 10, 12, 15) plus MiniLM k = 12 per corpus. Unforced Digital–Absence mix = **0**.

**Still human work**

1. Code 150 emotion windows (local pack, not in git: `annotation/emotion_windows_150.xlsx` / `.html`). Do not open “MODEL GUESSES” until all 150 rows are coded.
2. Write the paper. Prefer **domain-level** claims. Declared pair hypotheses mostly fail or split by method — report that honestly.

---

## What we measured

| Method | What it answers | Role in this study |
|--------|-----------------|--------------------|
| **PPMI** | Do local collocates of a domain look the same in A and B? | Strongest A/B *collocational* contrast |
| **BERT** (`bert-base-uncased`, frozen) | Do pretrained contextual senses land in the same region? | Stable neighbours; high cross-corpus similarity with a residual difference |
| **Word2Vec / FastText** | Do *trained-on-site* embeddings recover the same picture? | Profiles stable; neighbour lists weaker than BERT |
| **Emotion** (DistilRoBERTa, 7 classes) | Does local emotion around anchors differ? | Window proxy for aspect emotion, not trained ABSA |
| **Pooled NMF** | Shared topic space with bootstrap stability | k = 12, 11/12 core |
| **Independent LDA / NMF / MiniLM** | Do A and B grow the *same* topics when fitted separately? Unforced Digital–Absence? | Digital–Absence never co-cluster at k = 12 |

Shared vocabulary basis for comparable profiles: types with frequency ≥ 20 in **both** corpora (**2,286** dimensions).

---

## Sample and seed dictionary

| | Corpus A | Corpus B |
|--|---------:|---------:|
| Posts | 10,000 | 10,000 |
| Subreddits | expats 3,889; StudyAbroad 3,240; immigration 2,871 | GriefSupport 9,752; bereavement 248 |
| Primary files | [`data/analysis/corpus_a.csv`](data/analysis/corpus_a.csv) | [`data/analysis/corpus_b.csv`](data/analysis/corpus_b.csv) |

Rules: posts only; body > 100 words (title does not count); fail-closed dates; A keyword blacklist is whole-word (so `ead`/`irs` substring bugs do not fire). Prefer `data/analysis/` for new work. Raw dumps are not in git.

### Four seed domains (71 anchors)

| Domain | Label | Example terms | Posts hitting domain (full dict) | Strict (non-polysemous) |
|--------|--------|---------------|----------------------------------:|------------------------:|
| **Temporal** | Time–space disruption | timezone, weekend, schedule, distance, stuck | A **26.4%** · B **47.9%** | A 8.2% · B 12.8% |
| **Identity** | Symbolic order / belonging | name, accent, belong, culture shock, foreigner | A **16.8%** · B **16.2%** | A 11.4% · B 4.7% |
| **Absence** | Existential absence / grief | alone, lost, gone, missing, ghost, funeral | A **19.7%** · B **84.1%** | A 10.3% · B 58.7% |
| **Digital** | Digital contact | skype, screen, wifi, video call, connection | A **7.8%** · B **8.0%** | A **2.6%** · B 2.8% |

Absence dominates grief posts. Identity is denser in migration under the strict list. **Digital is the sparsest domain in both corpora** even after 10k.

![Domain coverage](docs/figures/p2_09_domain_coverage.png)

*Share of posts hitting each domain: full dictionary vs strict (non-polysemous) list.*

Usable **individual** anchors (frequency ≥ 20 in the weaker corpus **and** reliability gate): **29/71** PPMI, **40/71** BERT, **48/71** Word2Vec, **48/71** FastText. All **4/4 domains** are usable. Prefer domain-level claims.

---

## Phase 2 - semantics

We compare how each domain is represented in migration vs grief.  
**cos\*** = attenuation-corrected cosine (higher ≈ more similar across corpora).  
Significance: post-level resampling + Benjamini–Hochberg FDR.

| Domain | PPMI cos\* | PPMI A vs B different? | BERT cos\* | BERT residual difference? |
|--------|-----------:|:----------------------:|-----------:|:-------------------------:|
| Temporal | 0.71 | **Yes** (q = 0.0001) | 0.92 | Yes (q = 0.002) |
| Absence | 0.67 | **Yes** (q = 0.0001) | 0.93 | Yes (q = 0.002) |
| Identity | 0.70 | **Yes** (q = 0.0001) | 0.86 | Yes (q = 0.002) |
| Digital | 0.71 | **Yes** (q = 0.0001) | 0.97 | Yes (q = 0.002) |

**How to read this**

- **PPMI:** “Do the *local collocates* look the same?” **No** for all four domains (q = 0.0001). Digital is no longer the n.s. exception it was on the 906/687 pilot.
- **BERT:** “Do *pretrained senses* land in the same region?” **Mostly yes** (high cos\*), but still not interchangeable under a label-permutation null.
- These are **complementary**, not contradictory. PPMI is the discriminative collocational signal; BERT is the stable sense geometry.

### Domain profile geometry — PPMI

Eight points = 4 domains × 2 corpora. Colour = corpus.

**PCA**

![PPMI PCA](docs/figures/p2_18_domain_profiles_pca.png)

**t-SNE**

![PPMI t-SNE](docs/figures/p2_18_domain_profiles_tsne.png)

### Domain profile geometry — frozen BERT

**PCA**

![BERT PCA](docs/figures/p2_18_domain_profiles_pca__contextual.png)

**t-SNE**

![BERT t-SNE](docs/figures/p2_18_domain_profiles_tsne__contextual.png)

### PPMI vs BERT cos\*

Points above the diagonal: BERT finds *more* cross-corpus similarity than PPMI (true for every domain).

![cos* PPMI vs BERT](docs/figures/p2_23_cosstar_ppmi_vs_contextual.png)

### Word2Vec and FastText (trained on these texts)

Full 10k/10k evaluation (12 splits / 20 bootstrap / 30 permutations).

| Method | Domain profile reliability (mean A / B) | Neighbour Jaccard (mean A / B) | Domain cos\* |
|--------|-----------------------------------------|--------------------------------|--------------|
| BERT | 0.999 / 0.999 | **0.60 / 0.75** | 0.86–0.97 |
| FastText | 0.992 / 0.990 | 0.17 / 0.36 | 0.97–0.98 |
| Word2Vec | 0.988 / 0.986 | 0.13 / 0.29 | 0.96–0.98 |
| PPMI | 0.55 / 0.58 | 0.04 / 0.05 | 0.67–0.71 |

Chance neighbour Jaccard ≈ **0.0033** (basis = 2,286).

**Takeaway at 10k**

1. W2V/FT **domain profiles** are stable (rel ≈ 0.99).
2. Their **top-15 neighbour lists** beat chance but stay well below BERT — do not headline W2V/FT neighbours.
3. W2V/FT **cos\*** is near 1 (little A/B contrast). **PPMI** is most discriminative for collocates; **BERT** sits in between.

BERT split-half reliability is measured **without** per-half all-but-the-top (ABTT). ABTT on only four domain profiles per half destroyed reliability; ABTT is only used when there are enough points.

---

## Declared pair hypotheses

Four pre-declared domain-pair stories, scored on 10k with PPMI and BERT (CI on Δ = A − B). Full sheet: regenerate `outputs/tables/p2_14_hypothesis_outcomes.csv` locally.

| Comparison | Declared story | PPMI | BERT | Overall |
|------------|----------------|------|------|---------|
| Digital vs Absence | Closer in **migration** | not supported (CI includes 0) | **supported** | **partial** |
| Temporal vs Absence | Closer in **grief** | **opposite sign** | **opposite sign** | **rejected** |
| Identity vs Absence | Closer in **migration** | not supported | **supported** | **partial** |
| Digital vs Temporal | Closer in **migration** | **opposite sign** | **supported** | **methods disagree** |

Both methods put *temporal vs absence* **closer in migration**, opposite the brief. BERT and PPMI **disagree on the sign** of *digital vs temporal*. Report both methods; do not pick the convenient one.

---

## Phase 3 - emotion and co-occurrence

**Model:** `j-hartmann/emotion-english-distilroberta-base` (anger, disgust, fear, joy, neutral, sadness, surprise).  
**Design:** sentence-clipped ±5-token windows; one post per author for inference; polysemous anchors excluded from strict estimates; FDR across domain tests.

| Domain | Emotion distribution A vs B | FDR q | Notable A − B |
|--------|----------------------------|------:|----------------|
| Identity | **Different** | 0.0003 | sadness −0.17 (higher sadness in grief) |
| Absence | **Different** | 0.0003 | surprise higher in grief |
| Temporal | **Different** | 0.0003 | sadness higher in grief; more neutral in migration |
| Digital | **Different** | 0.0003 | **neutral +0.21** in migration |

This is a **window proxy** for aspect-based emotion (emotion of local context; aspect ≈ anchor), **not** a trained ABSA model. Human coding of 150 windows is still required before treating labels as validated.

**Stacked emotion mix by domain and corpus**

![Emotion stacked by domain](docs/figures/p3_19_emotion_stacked_domain.png)

**Emotion difference heatmap** (Corpus A − Corpus B; `*` survives multiple-testing correction)

![Emotion difference heatmap](docs/figures/p3_19_emotion_difference_heatmap.png)

### Co-occurrence networks

Top 300 PPMI edges. Exploratory: many edges have low raw counts.

**Corpus A — migration**

![Co-occurrence A](docs/figures/p3_20_cooccurrence__a.png)

**Corpus B — grief**

![Co-occurrence B](docs/figures/p3_20_cooccurrence__b.png)

---

## Phase 4 - topics

Two complementary designs.

1. **Pooled NMF** on A+B with bootstrap stability (shared topic space, comparable prevalence).
2. **Independent** LDA, NMF, and MiniLM on A and B *separately* (no forced alignment; tests whether Digital and Absence fall into one topic when unforced).

### Pooled NMF (stability-selected)

- TF–IDF NMF, bootstrap stability selection.
- k-sweep: k = 8 has the highest mean agreement (**0.96**); **k = 12** keeps **11/12** core topics and the highest agreement-above-null (**0.80**). Selected **k = 12**.

| Topic | Load | Recurrence | Core? | Top terms (short) |
|------:|------|-----------:|:-----:|-------------------|
| 0 | B — hospital days | 1.00 | ★ | day, week, time, hospital, cry, die |
| 1 | A — language/culture | 0.50 | | country, language, english, culture, foreigner |
| 2 | A — university | 1.00 | ★ | university, student, master, degree, scholarship |
| 3 | B — father loss | 1.00 | ★ | dad, die, mum, passing_away, parent, lost |
| 4 | A — passport / citizenship | 1.00 | ★ | passport, citizen, birth, immigration, esta |
| 5 | B — grief support | 0.80 | ★ | grief, lost, loss, help, grieve, support |
| 6 | B — mom / sister cancer | 1.00 | ★ | mom, cancer, sister, pass, grandma |
| 7 | A — study abroad | 1.00 | ★ | study, abroad, program, semester, scholarship |
| 8 | B — love / miss | 0.95 | ★ | love, miss, never, gone, without |
| 9 | B — family deaths | 0.95 | ★ | brother, mother, family, father, sister, die |
| 10 | Shared — friends/talk | 1.00 | ★ | friend, talk, meet, group, family |
| 11 | A — move / job | 0.85 | ★ | move, job, work, live, uk, family |

**Topic prevalence by corpus** (★ = core)

![Topic prevalence](docs/figures/p4_20_topic_prevalence.png)

**Topic stability** across bootstrap runs (dashed line = core threshold)

![Topic stability](docs/figures/p4_20_topic_stability.png)

### Independent models (per corpus)

Script: [`scripts/33_phase4_full_compare.py`](scripts/33_phase4_full_compare.py).

| Fit | Grid |
|-----|------|
| LDA, separately on A and on B | k = 10, 12, 15 |
| NMF, separately on A and on B | k = 10, 12, 15 |
| MiniLM-L6-v2 + KMeans + class-based TF-IDF | k = 12 (GPU if available) |

**Headline.** At k = 12, **zero topics mix Digital and Absence** on either corpus under LDA, NMF, or MiniLM. Digital anchors almost never enter the top-10. Absence appears in grief NMF (loss / family) as expected. Independent models do **not** force a Digital–Absence cluster.

**Independent NMF k = 12 — most prevalent topics**

| Corpus | Topic | Mean θ | Seed hits | Top terms |
|--------|------:|-------:|-----------|-----------|
| A | 2 | 0.13 | — | immigration, help, case, lawyer, ice, process |
| A | 3 | 0.10 | — | master, degree, job, work, bachelor, career |
| A | 5 | 0.10 | — | people, friend, meet, group, expat, social |
| A | 1 | 0.09 | — | study, abroad, program, college, semester, scholarship |
| B | 2 | 0.14 | — | tell, call, home, hospital, week, last |
| B | 4 | 0.13 | Absence | love, miss, never, wish, without, gone |
| B | 0 | 0.12 | Absence | grief, people, loss, help, lost, grieve |
| B | 3 | 0.08 | Absence | dad, parent, die, passing_away, family, lost |

**Independent LDA k = 12 — most prevalent topics**

| Corpus | Topic | Mean θ | Seed hits | Top terms |
|--------|------:|-------:|-----------|-----------|
| A | 1 | 0.20 | — | feel, go, like, want, move, back, home |
| A | 10 | 0.18 | — | study, university, student, abroad, program, school |
| A | 5 | 0.16 | — | work, job, move, uk, live, country |
| A | 3 | 0.11 | — | passport, citizen, citizenship, immigration, birth |
| B | 7 | 0.56 | Absence | dad, mom, friend, die, lost, family |
| B | 4 | 0.20 | — | love, never, miss, still, always, remember |
| B | 11 | 0.11 | — | hospital, tell, week, call, mom, home |

LDA on B is peakier (one family-loss topic takes ~56% mean θ). NMF spreads mass more evenly. MiniLM recovers finer clusters (homesickness vs deportation vs study logistics on A; ambulance/CPR vs bereavement resources on B) but does not create a Digital–Absence topic either.

**Corpus A — LDA k=12 topic-keyword network** (Identity / Time / Digital / Absence coloured)

![Topic network A](docs/figures/p4_33_topic_network__a.png)

**Corpus B — LDA k=12 topic-keyword network**

![Topic network B](docs/figures/p4_33_topic_network__b.png)

**Interactive topic views** (open on GitHub, then “View raw” / download if the preview is static):

| | Corpus A | Corpus B |
|--|----------|----------|
| LDA k=12 | [html](docs/figures/p4_33_interactive__lda__a.html) | [html](docs/figures/p4_33_interactive__lda__b.html) |
| NMF k=12 | [html](docs/figures/p4_33_interactive__nmf__a.html) | [html](docs/figures/p4_33_interactive__nmf__b.html) |
| MiniLM k=12 | [html](docs/figures/p4_33_interactive__minilm__a.html) | [html](docs/figures/p4_33_interactive__minilm__b.html) |

Numeric tables regenerate locally under `outputs/tables/p4_33_alignment.csv`, `p4_33_topic_terms.csv`, `p4_33_digital_absence_mix.csv`, `p4_33_domain_bridges.csv`.

---

## What holds vs what is weak

| Holds on 10k / 10k | Weak / incomplete |
|--------------------|-------------------|
| Documented 10k/10k sampling frame with quotas | Most individual anchors still too rare for inference |
| Domain-level PPMI contrast (4/4 FDR) | True ABSA not trained (window proxy only) |
| Frozen BERT neighbour stability | Manual coding of the 150-window sample |
| Emotion differences for all four domains | W2V/FT neighbours weaker than BERT |
| Pooled NMF 11/12 core topics at k=12 | BERT/PPMI disagree on the sign of digital vs temporal |
| Independent models: **no Digital–Absence mix** | Digital too sparse (~8% posts; ~2.6% strict) to surface as its own topic |
| Declared pair hypotheses scored honestly | Paper text not written yet |

**For the paper:** report PPMI and BERT together. Do not headline Word2Vec/FastText neighbours. Prefer domain-level claims. State that independent topic models do not recover a Digital–Absence cluster.

---

## Figure index

| Figure | File |
|--------|------|
| Domain coverage | [p2_09_domain_coverage.png](docs/figures/p2_09_domain_coverage.png) |
| PPMI domain PCA | [p2_18_domain_profiles_pca.png](docs/figures/p2_18_domain_profiles_pca.png) |
| PPMI domain t-SNE | [p2_18_domain_profiles_tsne.png](docs/figures/p2_18_domain_profiles_tsne.png) |
| BERT domain PCA | [p2_18_domain_profiles_pca__contextual.png](docs/figures/p2_18_domain_profiles_pca__contextual.png) |
| BERT domain t-SNE | [p2_18_domain_profiles_tsne__contextual.png](docs/figures/p2_18_domain_profiles_tsne__contextual.png) |
| cos\* PPMI vs BERT | [p2_23_cosstar_ppmi_vs_contextual.png](docs/figures/p2_23_cosstar_ppmi_vs_contextual.png) |
| Emotion stacked bars | [p3_19_emotion_stacked_domain.png](docs/figures/p3_19_emotion_stacked_domain.png) |
| Emotion difference heatmap | [p3_19_emotion_difference_heatmap.png](docs/figures/p3_19_emotion_difference_heatmap.png) |
| Co-occurrence A | [p3_20_cooccurrence__a.png](docs/figures/p3_20_cooccurrence__a.png) |
| Co-occurrence B | [p3_20_cooccurrence__b.png](docs/figures/p3_20_cooccurrence__b.png) |
| Topic prevalence (pooled NMF) | [p4_20_topic_prevalence.png](docs/figures/p4_20_topic_prevalence.png) |
| Topic stability (pooled NMF) | [p4_20_topic_stability.png](docs/figures/p4_20_topic_stability.png) |
| Independent topic network A | [p4_33_topic_network__a.png](docs/figures/p4_33_topic_network__a.png) |
| Independent topic network B | [p4_33_topic_network__b.png](docs/figures/p4_33_topic_network__b.png) |

Longer methods write-up: [`docs/METHODS_RESULTS.md`](docs/METHODS_RESULTS.md).

---

## Reproduce

### Repository layout

```
config/           filters, scrape plan, seed dictionary, analysis knobs
data/processed/   Phase 1 cleaned CSVs + PHASE1_COMPLETE.md
data/analysis/    primary research CSVs for Phases 2–4
docs/figures/     publication figures (this README)
docs/METHODS_RESULTS.md
scripts/          scrape → filter → analysis (incl. 33_phase4_full_compare.py)
tests/
run_analysis.py              Phase 2–4 PPMI default path
run_contextual_phase2.py     frozen BERT + methods join
run_predictive_phase2.py     Word2Vec / FastText evaluation
```

Python 3.12. Analysis uses a **separate** venv from scraping so torch does not disturb spaCy.

```powershell
python -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -U pip
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements.txt -r requirements-analysis.txt -c constraints-analysis.txt
.\.venv-analysis\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis-torch.txt -c constraints-analysis.txt
.\.venv-analysis\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv-analysis\Scripts\python.exe -m spacy download en_core_web_md
.\.venv-analysis\Scripts\python.exe scripts\check_env.py
.\.venv-analysis\Scripts\python.exe run_analysis.py
.\.venv-analysis\Scripts\python.exe run_contextual_phase2.py
.\.venv-analysis\Scripts\python.exe run_predictive_phase2.py
.\.venv-analysis\Scripts\python.exe scripts\33_phase4_full_compare.py
.\.venv-analysis\Scripts\python.exe -m pytest
```

Rebuild the raw archive only if needed (not required to analyse the CSVs already in this repo):

```powershell
python scripts/31_scrape_arctic_shift.py
python scripts/03_filter_and_normalize.py
python scripts/04_preprocess_nlp.py
python scripts/08_reprocess_tokens.py
```

Tables and extra figures land under `outputs/` locally (gitignored). Published figures live in `docs/figures/`.

Annotation pack (local only): `python scripts/22_annotation_sample.py --replace` then `python scripts/32_make_annotation_workbook.py`.

---

## Ethics

Reddit content is subject to Reddit’s terms and user privacy expectations. Use for research only. Do not redistribute personal data without appropriate review. This repository is private because the CSVs include post text, usernames, and URLs.
