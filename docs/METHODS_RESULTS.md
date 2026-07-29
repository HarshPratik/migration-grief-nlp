# Methods and results on the current corpora (no new scraping)

**Last updated:** 2026-07-29  
**Analysis corpora:** Corpus A (migration) **906** posts; Corpus B (grief) **687** posts (`data/analysis/`).  
**Seed dictionary:** 4 domains, **71** anchors (`config/anchors.yaml`, from `phases/Seed Dictionary.docx`).

This document records what was run against the phase briefs (except further scraping), why Word2Vec/fastText were not used, how PPMI and BERT were compared, and what each method shows.

---

## 1. Phase requirements → what we ran

| Phase doc | Requirement | Status on current data |
|-----------|-------------|------------------------|
| Phase 1 | Filter, clean, CSV A/B (scrape deferred) | **Done** — `data/interim`, `data/analysis` CSVs/parquets; frozen `data/processed` is an older A=323 snapshot |
| Phase 2 | Preprocess; embeddings; top-15 neighbors A/B; cosine A vs B; CSV/Excel; PCA/t-SNE | **Done** with **two** embedding methods (PPMI + BERT) |
| Phase 3 | Emotion on anchor windows; ABSA-style; stats; matrices; stacked bars/heatmaps | **Done** (window-proxy ABSA + DistilRoBERTa-emotion) |
| Seed dict | Four domains | **Done** |
| Topics (pipeline Phase 4) | Topic model | **Done** (pooled NMF, k=8 by stability) |

Word2Vec and FastText **are evaluated empirically** on these corpora (not dismissed without a run). They are trained from the research texts; reliability uses split-half **re-training** and second-order profiles over the shared basis (alignment-free). Frozen BERT still *measures* pretrained senses rather than estimating free parameters per type. See §2.4 and the multi-method scoreboard after `scripts/25_predictive_eval.py` / `scripts/23b_methods_dashboard.py`.

---

## 2. Representation methods (what we tried and why)

### 2.1 PPMI context profiles (count-based, alignment-free)

- Shared basis: types with frequency ≥ 20 in **both** corpora (**539** dimensions).
- For each anchor/domain in each corpus: window co-occurrence counts → PPMI → profile over the shared basis.
- Neighbours = top-15 basis dimensions by PPMI weight.
- **Primary unit:** domain (4/4 usable under reliability gate). **12/71** anchors usable under PPMI reliability+frequency floor.

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

**Significance agreement (usable FDR calls):** **10** both_yes · **62** both_no · **3** disagree.

### 2.4 Word2Vec and FastText — empirical evaluation (not assumed a priori)

Trained on the analysis corpora with gensim 4.4 (numpy 2 compatible). Settings: skip-gram, dim=100, window=5, min_count=5, epochs=20, workers=1. Evaluation matches the other methods:

- Split-half **re-training** (12 splits)
- Second-order profiles over the **same 539-type shared basis** (alignment-free)
- Top-15 neighbours restricted to that basis
- Bootstrap (20) + stratified permutation (30) for A/B comparison

Runner: `python run_predictive_phase2.py` or `python scripts/25_predictive_eval.py`.

#### Domain reliability scoreboard (measured)

| Method | Usable domains | Mean rel A / B | Mean neighbour Jaccard A / B | Chance Jaccard |
|--------|----------------|----------------|------------------------------|----------------|
| **BERT (contextual)** | 4/4 | **0.993 / 0.985** | **0.526 / 0.479** | 0.014 |
| FastText | 4/4 | 0.996 / 0.996 | 0.115 / 0.090 | 0.014 |
| Word2Vec | 4/4 | 0.988 / 0.988 | 0.123 / 0.062 | 0.014 |
| PPMI | 4/4 | 0.504 / 0.529 | 0.061 / 0.049 | 0.014 |

#### Domain cos* (A vs B similarity; higher = more similar)

| Domain | PPMI | BERT | Word2Vec | FastText |
|--------|-----:|-----:|---------:|---------:|
| Digital | 0.80 | 0.96 | 0.98 | 0.99 |
| Temporal | 0.72 | 0.91 | 0.98 | 0.99 |
| Absence | 0.60 | 0.91 | 0.98 | 0.99 |
| Identity | 0.58 | 0.82 | 0.98 | 0.99 |

#### What this means for the paper (honest reading)

1. **We did not skip Word2Vec/FastText.** They were trained and scored with the same reliability protocol.
2. **Second-order profiles are stable** for domain-pooled W2V/FT at this N (rel ≈ 0.99). So it is **false** to say “predictive embeddings produce pure noise here.”
3. **Neighbour lists are the weak part for W2V/FT:** mean Jaccard ~0.06–0.12 vs chance 0.014 — only modestly above chance, and **far below BERT** (~0.5). Phase 2’s requested “top-15 nearest neighbours” is therefore **not trustworthy** as a finding under W2V/FT at current size.
4. **cos* under W2V/FT is near 1.0** for all domains — little A/B contrast. **PPMI** shows the strongest collocational divergence; **BERT** is intermediate.
5. **FastText caveat:** subword vectors give high reliability even for near-zero frequency anchors; we still require `min_f ≥ 20` so those do not enter the FDR family.
6. **At ~10k posts/corpus**, re-run `scripts/25_predictive_eval.py` with the same settings. The prediction is that neighbour Jaccard rises; until then, the measured gap vs BERT is evidence, not assertion.

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
| Digital | 0.80 | No (q≈0.08) | **0.96** | Yes (q=0.002) |
| Temporal | 0.72 | **Yes** | **0.91** | Yes |
| Absence | 0.60 | **Yes** | **0.91** | Yes |
| Identity | 0.58 | **Yes** | **0.82** | Yes |

**How to read this:**

- **PPMI** asks: “Do the *local collocates* of these words look the same in A and B?” Answer: **often no** (especially Temporal, Absence, Identity). Digital collocates are noisier / less divergent after FDR.
- **BERT** asks: “Do the *contextual embeddings* of these words land in the same region of pretrained space?” Answer: **mostly yes** (very high cos*), but still **slightly lower** than a label-permutation null, so residual A/B structure remains.

These are complementary, not contradictory. For a paper: report both; do not claim Word2Vec-style “semantic change” without the reliability gate.

### 3.2 Declared domain-pair hypotheses

| Comparison | PPMI Δ(A−B) CI excludes 0? | BERT Δ(A−B) CI excludes 0? |
|------------|:--------------------------:|:--------------------------:|
| digital vs absence | Yes | Yes |
| temporal vs absence | Yes | Yes |
| identity vs absence | No | Yes |
| digital vs temporal | No | Yes |

BERT’s high within-domain reliability gives tighter domain-pair geometry; PPMI is more conservative on identity/digital pairs.

### 3.3 Per-seed top-15 neighbours

Long-form exports (backend, unit, target, corpus, rank, neighbor):

- `outputs/tables/p2_14_neighbors_top15__ppmi.csv` — **2250** rows  
- `outputs/tables/p2_14_neighbors_top15__contextual.csv` — **2250** rows  

Qualitatively, BERT neighbours are more thematically coherent (e.g. absence → *lonely, lose, depressed, grief*). PPMI neighbours are corpus-specific collocates and often look less “semantic” but are the honest second-order signal.

### 3.4 Usable anchors (frequency floor + reliability)

| Backend | Domain usable | Anchor usable (min rel ≥ 0.25 and min f ≥ 20) |
|---------|---------------|-----------------------------------------------|
| PPMI | 4/4 | 12/71 |
| BERT | 4/4 | 8/71 |

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
| Domain Identity | **q ≈ 0.006** different A vs B |
| Domain Absence | **q ≈ 0.0002** |
| Domain Temporal | **q ≈ 0.0003** |
| Domain Digital | **q ≈ 0.80** not significant |
| ABSA | Window **proxy** (emotion of local context; aspect = anchor) — not a trained ABSA model |
| Networks | PPMI co-occurrence, top 300 edges, GraphML + plots |
| Annotation sample | 150 windows prepared (manual coding still human work) |

Figures: `p3_19_emotion_stacked_domain.png`, `p3_19_emotion_difference_heatmap.png`, `p3_20_cooccurrence__{a,b}.png`.

---

## 5. Phase 4 results (topics)

| Item | Result |
|------|--------|
| Method | Pooled NMF, TF-IDF, stability selection |
| Selected k | **8** (best baseline-adjusted stability) |
| Core topics (recurrence ≥ 0.8) | **Only Topic 2** — study-abroad language (higher in A) |
| Other topics | Grief/hospital, homesick, job/move, family loss — present but **unstable** across restarts |

Figures: `p4_20_topic_prevalence.png`, `p4_20_topic_stability.png`.

---

## 6. What works vs what does not (for the write-up)

### Works well on this sample size

1. **Domain-level** analysis (4 seed domains) under both PPMI and BERT.  
2. **Frozen BERT** sense vectors with document-level caching (stable split-half reliability).  
3. **PPMI** collocational contrast (clearer A/B *divergence* in cos*).  
4. **Emotion window proxy** with author clustering + FDR (Identity / Absence / Temporal).  
5. **NMF with stability** (honest: only one core topic).  
6. **Reliability gates** — prevents publishing neighbour lists as findings when they are chance-level.

### Does not work / weak

1. **Word2Vec / FastText / SGNS trained on these corpora** — underpowered; not used.  
2. **Most individual anchors** — rare in one or both corpora; only ~8–12 clear the joint frequency+reliability gate.  
3. **Digital domain** for emotion — not significant after strict methods.  
4. **Topic structure** — mostly unstable; interpret only core Topic 2 with caution.  
5. **True ABSA** — not trained; window proxy only.  
6. **Volume** — far below 10k/corpus research target (scraping intentionally paused).

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
outputs/tables/p4_17_*
outputs/figures/p2_18_* p2_23_* p3_19_* p3_20_* p4_20_*
outputs/networks/*.graphml
docs/METHODS_RESULTS.md   ← this file
run_analysis.py
run_contextual_phase2.py
```

---

## 9. Bottom line for hypotheses

On the **existing** data, without further scraping:

- Seed-domain **semantic organisation** differs between migration and grief under **PPMI** for Temporal, Identity, Absence; Digital is borderline.  
- Under **BERT**, the same domains are **highly similar** across corpora but still not interchangeable under a permutation null; domain-pair geometry is stronger.  
- **Emotion** around anchors differs for Identity, Absence, Temporal — not Digital.  
- **Topics** only robustly recover a Study Abroad theme in A.  
- Showing **both** PPMI and BERT (plus the explicit rejection of Word2Vec/fastText) is the correct small-data methodology story for the paper.
