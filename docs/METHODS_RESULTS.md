# Methods and results on the current corpora (no new scraping)

**Last updated:** 2026-08-30  
**Analysis corpora:** Corpus A (migration) **10,000** posts; Corpus B (grief) **10,000** posts (`data/analysis/`).  
**Seed dictionary:** 4 domains, **71** anchors (`config/anchors.yaml`, from `phases/Seed Dictionary.docx`).  
**Sampling:** Arctic Shift rebuild, seed `20260827`, window `2023-07-01 ≤ created_at < 2026-08-24`. See `data/processed/PHASE1_COMPLETE.md`.

This document records what was run against the phase briefs on the 10k/10k sample. Word2Vec/FastText full evaluation finished 2026-08-28.

---

## 1. Phase requirements → what we ran

| Phase doc | Requirement | Status on current data |
|-----------|-------------|------------------------|
| Phase 1 | Filter, clean, CSV A/B | **Done** — 10k/10k in `data/analysis`; audit in `data/processed/PHASE1_COMPLETE.md` |
| Phase 2 | Preprocess; embeddings; top-15 neighbors A/B; cosine A vs B; CSV/Excel; PCA/t-SNE | **Done** (PPMI, BERT, Word2Vec, FastText) |
| Phase 3 | Emotion on anchor windows; ABSA-style; stats; matrices; stacked bars/heatmaps | **Done** (window-proxy ABSA + DistilRoBERTa-emotion) |
| Seed dict | Four domains | **Done** |
| Topics (pipeline Phase 4) | Topic model | **Done** (pooled NMF k=12, 11/12 core; independent LDA/NMF k=10,12,15; MiniLM k=12) |

Word2Vec and FastText **are evaluated empirically** on these corpora (not dismissed without a run). They are trained from the research texts; reliability uses split-half **re-training** and second-order profiles over the shared basis (alignment-free). Frozen BERT still *measures* pretrained senses rather than estimating free parameters per type. See §2.4 and the multi-method scoreboard after `scripts/25_predictive_eval.py` / `scripts/23b_methods_dashboard.py`.

---

## 2. Representation methods (what we tried and why)

### 2.1 PPMI context profiles (count-based, alignment-free)

- Shared basis: types with frequency ≥ 20 in **both** corpora (**2,286** dimensions).
- For each anchor/domain in each corpus: window co-occurrence counts → PPMI → profile over the shared basis.
- Neighbours = top-15 basis dimensions by PPMI weight.
- **Primary unit:** domain (4/4 usable under reliability gate). Anchor usable count on 10k is in `p2_13_reliability__anchor.csv`; prefer domain-level claims.

**Outputs:** `data/analysis/vectors/*_ppmi.npz`, `outputs/tables/p2_13_*`, `p2_14_*` (no `__contextual` suffix).

### 2.2 Contextual BERT sense vectors (frozen `bert-base-uncased`)

- Encode each document once; mean of last 4 hidden layers; mean-pool wordpieces per whitespace token.
- Cache only tokens in anchors ∪ shared basis under `data/analysis/vectors/contextual_cache/`.
- Sense vector = mean of all cached occurrences of the target terms in that corpus.
- Neighbours = cosine of the sense vector to mean type embeddings of shared-basis words in that corpus.
- Split-half / permutation / bootstrap **never re-encode** — they re-average under document masks.

**Reliability note (important methodological result):** applying Mu & Viswanath all-but-the-top PCA on only **4** domain profiles *per half* destroyed domain reliability (cos≈0). Split-half reliability is measured **without** per-half ABTT; ABTT is only used when there are enough points (e.g. many type embeddings). After the fix, domain BERT reliability is **~0.96–0.999**.

**Outputs:** `*_contextual.npz`, `p2_13_*__contextual.csv`, `p2_14_*__contextual.csv`, long-form `p2_14_neighbors_top15__contextual.csv`.

### 2.3 Methods comparison package (PPMI vs BERT)

| File | Content |
|------|---------|
| `outputs/tables/p2_23_methods_reliability_compare.csv` | Side-by-side reliability |
| `outputs/tables/p2_23_methods_semantic_compare.csv` | Side-by-side cos / cos* / p |
| `outputs/tables/p2_23_methods_agreement.csv` | both_yes / both_no / disagree |
| `outputs/figures/p2_23_cosstar_ppmi_vs_contextual.png` | cos* scatter |
| `outputs/summaries/p2_23_methods_compare.json` | Counts |

**Significance agreement (usable FDR calls on 10k/10k):** **31** both_yes · **32** both_no · **12** disagree.

### 2.4 Word2Vec and FastText — empirical evaluation (not assumed a priori)

Trained on the analysis corpora with gensim 4.4 (numpy 2 compatible). Settings: skip-gram, dim=100, window=5, min_count=5, epochs=20, workers=1. Evaluation matches the other methods:

- Split-half **re-training** (12 splits)
- Second-order profiles over the **same 2,286-type shared basis** (alignment-free)
- Top-15 neighbours restricted to that basis
- Bootstrap (20) + stratified permutation (30) for A/B comparison

Runner: `python run_predictive_phase2.py` or `python scripts/25_predictive_eval.py`.

#### Domain reliability scoreboard (10k/10k)

| Method | Usable domains | Mean rel A / B | Mean neighbour Jaccard A / B | Chance Jaccard |
|--------|----------------|----------------|------------------------------|----------------|
| **BERT (contextual)** | 4/4 | **0.999 / 0.999** | **0.604 / 0.746** | 0.0033 |
| FastText | 4/4 | 0.992 / 0.990 | 0.171 / 0.360 | 0.0033 |
| Word2Vec | 4/4 | 0.988 / 0.986 | 0.130 / 0.287 | 0.0033 |
| PPMI | 4/4 | 0.548 / 0.577 | 0.042 / 0.053 | 0.0033 |

#### Domain cos* (A vs B similarity; higher = more similar)

| Domain | PPMI | BERT | Word2Vec | FastText |
|--------|-----:|-----:|---------:|---------:|
| Digital | 0.71 | 0.97 | 0.96 | 0.97 |
| Temporal | 0.71 | 0.92 | 0.98 | 0.98 |
| Absence | 0.67 | 0.93 | 0.97 | 0.98 |
| Identity | 0.70 | 0.86 | 0.96 | 0.97 |

#### What this means for the paper (honest reading)

1. **We did not skip Word2Vec/FastText.** They were trained and scored with the same reliability protocol.
2. **Second-order profiles are stable** for domain-pooled W2V/FT at this N (rel ≈ 0.99). So it is **false** to say “predictive embeddings produce pure noise here.”
3. **Neighbour lists are the weak part for W2V/FT:** mean Jaccard ~0.13–0.36 vs chance 0.0033 — above chance, and **far below BERT** (~0.60 / 0.75). Phase 2’s requested “top-15 nearest neighbours” is therefore **not the primary finding** under W2V/FT.
4. **cos* under W2V/FT is near 1.0** for all domains — little A/B contrast. **PPMI** shows the strongest collocational divergence; **BERT** is intermediate.
5. **FastText caveat:** subword vectors give high reliability even for near-zero frequency anchors; we still require `min_f ≥ 20` so those do not enter the FDR family.
6. **The 10k re-run is done** (`scripts/25_predictive_eval.py --algorithms word2vec fasttext --unit both --train-workers 8`). Neighbour Jaccard rose vs the 906/687 sample but remains far below BERT.

Multi-method tables:

| File | Content |
|------|---------|
| `outputs/tables/p2_23_reliability_scoreboard_domain.csv` | Domain scoreboard |
| `outputs/tables/p2_23_domain_min_reliability_by_method.csv` | Wide min-reliability |
| `outputs/tables/p2_23_domain_cosstar_by_method.csv` | Wide cos* |
| `outputs/tables/p2_23_all_methods_reliability.csv` | Stacked all backends |
| `outputs/tables/p2_23_all_methods_semantic.csv` | Stacked all backends |

---

## 3. Phase 2 results

### 3.1 Domain-level A vs B similarity

| Domain | PPMI cos* | PPMI sig (FDR) | BERT cos* | BERT sig (FDR) |
|--------|----------:|:--------------:|----------:|:--------------:|
| Digital | 0.71 | **Yes** (q=0.0001) | **0.97** | Yes (q=0.002) |
| Temporal | 0.71 | **Yes** (q=0.0001) | **0.92** | Yes (q=0.002) |
| Absence | 0.67 | **Yes** (q=0.0001) | **0.93** | Yes (q=0.002) |
| Identity | 0.70 | **Yes** (q=0.0001) | **0.86** | Yes (q=0.002) |

**How to read this:**

- **PPMI** asks: “Do the *local collocates* of these words look the same in A and B?” Answer: **no for all four domains** after FDR (q=0.0001). Digital is no longer the exception it was on the 906/687 sample.
- **BERT** asks: “Do the *contextual embeddings* of these words land in the same region of pretrained space?” Answer: **mostly yes** (very high cos*), but still **slightly lower** than a label-permutation null, so residual A/B structure remains.

These are complementary, not contradictory. For a paper: report both; do not claim Word2Vec-style “semantic change” without the reliability gate.

### 3.2 Declared domain-pair hypotheses

| Comparison | PPMI Δ(A−B) CI excludes 0? | BERT Δ(A−B) CI excludes 0? |
|------------|:--------------------------:|:--------------------------:|
| digital vs absence | No (diff +0.024) | **Yes** (diff +0.024, CI +0.014 to +0.034) |
| temporal vs absence | Yes (diff +0.081) | **Yes** (diff +0.100, CI +0.093 to +0.106) |
| identity vs absence | No (diff +0.003) | **Yes** (diff +0.021, CI +0.011 to +0.031) |
| digital vs temporal | Yes (diff −0.117) | **Yes** (diff +0.038, CI +0.028 to +0.049) |

PPMI pair CIs: `p2_14_domain_pair_similarity.csv`. BERT pair CIs: `p2_14_domain_pair_similarity__contextual.csv` (refreshed 2026-08-28 `--unit domain`). Supported/rejected calls: `p2_14_hypothesis_outcomes.csv`. BERT finds all four pair differences; PPMI is more conservative on digital/identity vs absence. The declared *temporal vs absence* story (closer in grief) is **not** supported: both methods put temporal–absence closer in **migration** (positive A−B). BERT *digital vs temporal* is also the opposite sign from PPMI.

### 3.3 Per-seed top-15 neighbours

Long-form exports (backend, unit, target, corpus, rank, neighbor):

- `outputs/tables/p2_14_neighbors_top15__ppmi.csv` — **2250** rows  
- `outputs/tables/p2_14_neighbors_top15__contextual.csv` — **2250** rows  

Qualitatively, BERT neighbours are more thematically coherent (e.g. absence → *lonely, lose, depressed, grief*). PPMI neighbours are corpus-specific collocates and often look less “semantic” but are the honest second-order signal.

### 3.4 Usable anchors (frequency floor + reliability)

| Backend | Domain usable | Anchor usable (min rel ≥ 0.25 and min f ≥ 20) |
|---------|---------------|-----------------------------------------------|
| PPMI | 4/4 | 29/71 |
| BERT | 4/4 | 40/71 |
| Word2Vec | 4/4 | 48/71 |
| FastText | 4/4 | 48/71 |

BERT often has **higher raw reliability** but the **same frequency floor** still excludes rare Digital/Identity terms that barely appear in one corpus.

### 3.5 Figures

| Method | PCA / t-SNE |
|--------|-------------|
| PPMI | `p2_18_domain_profiles_pca.png`, `…_tsne.png` |
| BERT | `p2_18_domain_profiles_pca__contextual.png`, `…_tsne__contextual.png` |

---

## 4. Phase 3 results (emotion / co-occurrence)

Already complete on the analysis corpora; not re-scraped.

| Item | Result |
|------|--------|
| Model | `j-hartmann/emotion-english-distilroberta-base` (7 classes) |
| Design | ±5 token windows; author-capped; polysemy-strict for inference; BH-FDR |
| Domain Identity | **q = 0.0003** different A vs B (sadness A−B = −0.17) |
| Domain Absence | **q = 0.0003** |
| Domain Temporal | **q = 0.0003** |
| Domain Digital | **q = 0.0003** (neutral A−B = +0.21) |
| ABSA | Window **proxy** (emotion of local context; aspect = anchor) — not a trained ABSA model |
| Networks | PPMI co-occurrence, top 300 edges, GraphML + plots |
| Annotation sample | 150 windows prepared (manual coding still human work) |

Figures: `p3_19_emotion_stacked_domain.png`, `p3_19_emotion_difference_heatmap.png`, `p3_20_cooccurrence__{a,b}.png`.

---

## 5. Phase 4 results (topics)

### 5.1 Pooled NMF (stability-selected)

| Item | Result |
|------|--------|
| Method | Pooled NMF, TF-IDF, stability selection |
| Selected k | **12** (k=8 has higher mean agreement 0.96; k=12 keeps 11/12 core and higher agreement-above-null) |
| Core topics (recurrence ≥ 0.8) | **11/12** — A: university, study-abroad, language/culture, passport, move/job; B: father loss, cancer, grief support, love-miss, family deaths, hospital days; shared: friends |
| Other topics | Topic 1 (language/culture) is the only non-core at k=12 (recurrence 0.5) |

Figures: `p4_20_topic_prevalence.png`, `p4_20_topic_stability.png`.

### 5.2 Independent models (per corpus)

The Phase 4 brief also asks for **independent** topic models on A and B (not a pooled fit), 10–15 topics, unforced Digital–Absence clustering, keyword co-occurrence, interactive views, and Identity–Time–Digital bridges.

| Item | Result |
|------|--------|
| Script | `scripts/33_phase4_full_compare.py` |
| LDA / NMF | Fitted **separately** on A and on B at **k = 10, 12, 15** |
| MiniLM | `all-MiniLM-L6-v2` embeddings + KMeans + class-based TF-IDF at **k = 12** (CUDA) |
| Digital–Absence mix at k=12 | **0** topics for LDA, NMF, and MiniLM on both corpora |
| Digital in top-10 | Almost never; Digital is too sparse to surface as its own topic |
| Absence | Appears in Corpus B NMF (grief/loss/family), not mixed with Digital |
| Networks | LDA k=12 topic-keyword GraphML + PNG; Identity / Time / Digital / Absence bridges in `p4_33_domain_bridges.csv` |
| Interactive | `docs/figures/p4_33_interactive__{lda,nmf,minilm}__{a,b}.html` |

Independent models do **not** force Digital and Absence into one cluster. That is the honest result of leaving the clustering unforced.

Figures: `p4_33_topic_network__{a,b}.png`. Tables regenerate under `outputs/tables/p4_33_*`.

---

## 6. What works vs what does not (for the write-up)

### Works well on this sample size

1. **Domain-level** analysis (4 seed domains) under both PPMI and BERT.  
2. **Frozen BERT** sense vectors with document-level caching (stable split-half reliability).  
3. **PPMI** collocational contrast (clearer A/B *divergence* in cos*).  
4. **Emotion window proxy** with author clustering + FDR (all four domains, q=0.0003).  
5. **NMF with stability** (honest: 11/12 core at k=12). Independent LDA/NMF/MiniLM agree that Digital–Absence does not cluster together when unforced.  
6. **Reliability gates** — prevents publishing neighbour lists as findings when they are chance-level.

### Does not work / weak

1. **Word2Vec / FastText neighbour lists** — above chance but far below BERT; do not treat W2V/FT top-15 as the primary finding.  
2. **Most individual anchors** — still rare in one corpus; prefer domain-level claims.  
3. **Digital domain** remains the sparsest (~8% of posts) even though emotion and PPMI now clear FDR.  
4. **Topic 1** (language/culture) is the only non-core topic at k=12.  
5. **True ABSA** — not trained; window proxy only.  
6. **Declared pair hypotheses vs 10k CIs** — BERT finds all four pair differences, but *temporal vs absence* is closer in migration, not grief; BERT and PPMI disagree on the sign of *digital vs temporal*.

### Methodological lesson (BERT ABTT)

All-but-the-top anisotropy correction is unsafe when the profile set is tiny (4 domains). Measuring reliability on raw mean-pooled senses, and restricting ABTT to large type sets, is required for honest numbers.

---

## 7. How to reproduce

### Full Phase 2–4 (PPMI default chain)

```powershell
cd D:\research_paper
.\.venv-analysis\Scripts\python.exe run_analysis.py
```

### BERT Phase 2 + methods compare (after PPMI tables exist)

```powershell
.\.venv-analysis\Scripts\python.exe run_contextual_phase2.py
# or from reliability if cache already built:
.\.venv-analysis\Scripts\python.exe run_contextual_phase2.py --from-step reliability
```

### Independent Phase 4 topic comparison

```powershell
.\.venv-analysis\Scripts\python.exe scripts\33_phase4_full_compare.py
```

### Neighbour long-form from comparison CSVs

```powershell
.\.venv-analysis\Scripts\python.exe scripts\14b_export_neighbors.py
```

### Shareable archive

Only **`outputs/research_analysis_bundle_reviewed.zip`** is approved for sharing (aggregate-only). Rebuild with `scripts/21_export_bundle.py --replace` after regenerating tables if you want BERT CSVs inside a new bundle (current reviewed zip may predate BERT exports — keep BERT tables under `outputs/tables/` for the paper appendix either way).

---

## 8. File map (main deliverables)

```
config/anchors.yaml, analysis.yaml, filters.yaml
data/analysis/corpus_{a,b}.parquet
data/analysis/vectors/*_ppmi.npz
data/analysis/vectors/*_contextual.npz
data/analysis/vectors/contextual_cache/
outputs/tables/p2_09_* … p2_14_* … p2_23_*
outputs/tables/p3_11_* p3_15_* p3_16_* p3_22_*
outputs/tables/p4_17_* p4_33_*
outputs/figures/p2_18_* p2_23_* p3_19_* p3_20_* p4_20_* p4_33_*
outputs/networks/*.graphml
scripts/33_phase4_full_compare.py
docs/METHODS_RESULTS.md   ← this file
run_analysis.py
run_contextual_phase2.py
```

---

## 9. Bottom line for hypotheses

On the **10k/10k** sample (rebuild 2026-08-27; no further scraping needed):

- **PPMI:** all four seed domains **diverge** between migration and grief after FDR (q = 0.0001). Digital is no longer the borderline case it was on the 906/687 sample.  
- **BERT:** the same domains are **highly similar** across corpora (cos\* 0.86–0.97) but still not interchangeable under a permutation null (q = 0.002).  
- **Declared domain-pair hypotheses** mostly **fail or split by method.** Both PPMI and BERT put *temporal vs absence* closer in **migration**, opposite the brief. BERT and PPMI **disagree on the sign** of *digital vs temporal*. See `outputs/tables/p2_14_hypothesis_outcomes.csv`.  
- **Emotion** around anchors differs for **all four** domains (q = 0.0003), including Digital (migration windows more neutral). This is a window proxy, not trained ABSA.  
- **Topics:** pooled NMF **k = 12** keeps **11/12** core themes (A: university / study-abroad / passport / move-job; B: family death / cancer / grief support; shared: friends). k = 8 is more stable if you want fewer, cleaner cores. Independent LDA/NMF (k=10,12,15) and MiniLM k=12 find **no Digital–Absence mixed topic** on either corpus.  
- Report **PPMI and BERT together**. Word2Vec/FastText were run: domain profiles are stable, neighbour lists are the weak finding — do not headline them.

Digital coverage is still sparse (~8% of posts; **~2.6%** if polysemous anchors such as *connection* are dropped). Prefer **domain-level** claims; most individual seeds remain rare.
