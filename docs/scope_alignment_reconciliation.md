# Reconciliation — actual G4-off run vs reconstructed failure analysis

Analysis only. No production file was created or modified; nothing committed.
This is the only file this task created.

**Verdict: the discrepancy is retrieval nondeterminism, on 2 of 504 claims. It
is not a reconstruction error and not a gate-selection difference. One case
(claim 420) must be dropped from the scope-alignment analysis; its headline
conclusions are unaffected.**

---

## 1. Artifact located

`eval/results_20261011_160640_n504_section2_w0.05_prosactive_gatesg1g2g3.json`
is **present** in `eval/` (it was absent when the previous turn searched, and
appeared between turns). Not in any commit on any branch — untracked working-tree
file.

| metadata | value |
|---|---|
| n_claims | 504 |
| evidence_mode / max_sentences / section_weight | section / 2 / 0.05 |
| prosecution_mode | active |
| **adjudication_gates** | **`['g1', 'g2', 'g3']`** ✓ |
| timestamp / runtime | 2026-10-11T16:06:40Z / 206.4 s |
| metrics.accuracy | 84.72% (427/504) |

The recorded gate selection confirms the run used G1+G2+G3, so the configuration
is not the source of the discrepancy.

## 2. Exact override ID sets

**24 rescues (gold CONTRADICT)** — actual run:

```
10, 46, 83, 85, 89, 93, 101, 110, 122, 180, 204, 236, 262, 294,
317, 324, 341, 370, 374, 403, 412, 434, 449, 494
```

**18 regressions (gold SUPPORT)** — actual run:

```
2, 55, 56, 69, 88, 95, 150, 179, 285, 306, 338, 356, 368, 391, 418, 419, 452, 457
```

Total overrides: **42** = 24 + 18. ✓

## 3. Comparison with the reconstruction

| | reconstructed | actual | result |
|---|---|---|---|
| rescue IDs | 24 | 24 | **exactly identical, element for element** |
| regression IDs | 19 | 18 | actual is a strict subset |
| extra in reconstruction | — | — | **claim 420** |
| in actual but not reconstructed | — | — | none |

**The extra regression is claim 420** — "The DESMOND program demonstrates no
significant impact on weight loss."

## 4. Cause: retrieval nondeterminism

### Claim 420, side by side

| field | default run (2026-10-09) | G4-off run (2026-10-11) |
|---|---|---|
| `evidence_doc_id` | **40949706** | **11360768** |
| primary evidence | *"The groups differed on weight lost at 24 months. Surgery generates substantial weight loss…"* | *"RESULTS Significant treatment effects were found for healthy eating (47.5% versus 34.3%), exercise…"* |
| `candidate_pool_size` | 20 | 22 |
| `counter_evidence_doc_id` | 6841927 | 11360768 |
| counter section | conclusions | results |
| primary→counter P(contradiction) | **0.0084** | **0.6694** |
| **G2 (pool_coherent)** | **True** | **False** |
| adjudication status | TRUE_CONTRADICTION | NOT_CONTRADICTION |
| override applied | True | **False** |

**The pipeline retrieved a different document.** That changed the primary
evidence, enlarged the candidate pool from 20 to 22 sentences, changed which
sentence the Prosecutor selected as counter-evidence, and so changed the text G2
was evaluating. G2 then failed, because the new primary evidence contradicts the
new counter-evidence (P(contradiction) 0.669 → `CONTRADICTED`).

So this is **not** a gate flipping on identical inputs. Both runs evaluated G2
correctly; they evaluated it on *different sentences*.

### Scale of the instability — all 504 claims compared

| field | claims differing | % |
|---|---|---|
| `claim`, `gold` | 0 | 0.00% |
| **`baseline_verdict`** | **0** | **0.00%** |
| **`baseline_prediction`** | **0** | **0.00%** |
| `evidence_doc_id` (retrieved doc) | **2** | **0.40%** |
| `evidence` (primary premise) | 2 | 0.40% |
| `candidate_pool_size` | 2 | 0.40% |
| `counter_evidence` | 1 | 0.20% |
| `counter_section` | 1 | 0.20% |
| `verdict` | 20 | 3.97% |

Retrieval differed on exactly **two** claims:

| claim | doc (default → G4-off) | pool | baseline verdict | gold | override impact |
|---|---|---|---|---|---|
| **420** | 40949706 → 11360768 | 20 → 22 | PARTIALLY_SUPPORTED (both) | SUPPORT | **changed** — the whole discrepancy |
| **427** | 1649738 → 13910150 | 9 → 10 | PARTIALLY_SUPPORTED (both) | SUPPORT | none (not overridden in either run) |

The 20 `verdict` differences are **exactly** the symmetric difference of the two
override sets (19 newly overridden with G4 off, plus claim 420 no longer
overridden). **No verdict changed for any other reason** — verified by set
subtraction, which leaves the empty set.

### Classification of the cause

| candidate cause | verdict | evidence |
|---|---|---|
| artifact reconstruction error | **ruled out** | The reconstruction was correct *given the default run's retrieval*. All 24 rescues and 18 of 19 regressions match exactly; the override rule and the `baseline_prediction == "SUPPORT"` ⟺ `primary_verdict != CONTRADICTED` equivalence both held. |
| gate-selection difference | **ruled out** | Artifact metadata records `['g1','g2','g3']`; every one of the 42 overrides passed G1, G2 and G3. |
| **retrieval nondeterminism** | **CONFIRMED** | 2/504 claims retrieved different documents, with different primary evidence and pool sizes. Claim 420's G2 failure follows directly. |
| other pipeline difference | **ruled out for the baseline** | All 504 `baseline_verdict` and `baseline_prediction` values are identical, and arm D scores 421/504 (C 115/173, S 306/331) in *both* runs. Extraction and verification are stable given the same retrieval. |

### Most likely mechanism — hypothesis, not established

`HybridRetriever._build_semantic_index` calls
`chromadb.Client().create_collection(name="scifact_collection")` with no HNSW
parameters and **no seed**. HNSW graph construction assigns node levels
randomly, so two index builds over the identical corpus can return different
approximate top-*k* neighbours. With `semantic_k=3`, `bm25_k=3` and `top_n=2`,
a single swap inside the semantic top-3 can change the final reranked top-2.

That is consistent with everything observed — a small, claim-specific effect
that touches retrieval only and leaves BM25, extraction and verification
untouched. It is **not verified**: confirming it needs the corpus (absent here),
by building the index twice in one process and diffing query results for claims
420 and 427.

An alternative that cannot be excluded on this evidence: float nondeterminism in
the CrossEncoder flipping a near-tie in the rerank sort. Both predict exactly
this signature, and distinguishing them requires the Colab environment.

## 5. Actual run metrics (for the record)

| metric | arm D | arm F default (G1–G4) | **arm F G4-off (G1+G2+G3)** | pre-registered prediction |
|---|---|---|---|---|
| overall | 83.53% (421/504) | 83.93% | **84.72% (427/504)** | 84.52% |
| CONTRADICT | 66.47% (115/173) | 73.99% | **80.35% (139/173)** | 80.35% ✓ |
| SUPPORT | 92.45% (306/331) | 89.12% | **87.01% (288/331)** | 86.71% |
| overrides | — | 24 | **42** | 43 |
| rescues | — | 13 | **24** | 24 ✓ |
| regressions | — | 11 | **18** | 19 |
| override precision | — | 54.17% | **57.14%** | 55.81% |
| net claims | — | +2 | **+6** | +5 |

The D1 reconstruction was correct on every quantity that does not depend on
claim 420: rescues (24) and CONTRADICT accuracy (80.35%) matched exactly. The
single retrieval difference accounts for the whole deviation, and it moved the
result **in arm F's favour** — one fewer regression than predicted.

## 6. Effect on the scope-alignment failure analysis

`docs/scope_alignment_failure_analysis.{md,json}` were **not modified**, per the
brief. The required correction is recorded here instead.

**Claim 420 must be dropped** from the regression set (19 → 18). Its labels were
*intervention mismatch* and *outcome mismatch*. Corrected counts:

| mismatch type | as published (n=19) | corrected (n=18) |
|---|---|---|
| population mismatch | 7 (36.8%) | **7 (38.9%)** |
| intervention mismatch | 8 (42.1%) | **7 (38.9%)** |
| outcome mismatch | 8 (42.1%) | **7 (38.9%)** |
| evidence granularity mismatch | 5 (26.3%) | 5 (27.8%) |
| endpoint mismatch | 2 (10.5%) | 2 (11.1%) |
| direction mismatch | 2 (10.5%) | 2 (11.1%) |
| context/condition mismatch | 2 (10.5%) | 2 (11.1%) |
| causal/relation mismatch | 2 (10.5%) | 2 (11.1%) |
| ambiguous/other | 2 (10.5%) | 2 (11.1%) |
| temporal mismatch | 0 | 0 |

**The headline separation survives unchanged:**

| criterion | regressions (n=18) | rescues (n=24) |
|---|---|---|
| ≥1 non-direction dimension mismatch | **18 (100.0%)** | **2 (8.3%)** |
| direction inversion on a shared outcome | **2 (11.1%)** | **24 (100.0%)** |

The three most frequent axes remain population / intervention / outcome, now
tied at 38.9% each. The rescue-side analysis is entirely unaffected, since all
24 rescue IDs were correct.

## 7. Consequences worth acting on

1. **The pipeline is not reproducible at the per-claim level.** 2/504 claims
   (0.40%) retrieve different documents between runs. Aggregate arm comparisons
   are safe — arm D scored 421/504 identically in both runs, and all 504 baseline
   verdicts match — but **per-claim case lists are not stable across runs**, and
   1 of 42 overrides (2.4%) changed as a result. Any analysis that names specific
   claim IDs should cite the artifact it came from.

2. **Seeding retrieval would remove this.** Passing explicit HNSW parameters or
   a seed to `create_collection`, or replacing the approximate index with exact
   search over 5,183 documents (small enough to be cheap), would make retrieval
   deterministic. That is a production change and is **not** made here.

3. **The blind re-annotation in the failure analysis's §7 should use the 42-case
   set from this artifact** — the 24 rescue IDs and 18 regression IDs in §2 —
   not the 43-case reconstructed set.

4. **D1's reconstruction method is validated.** It predicted 24/24 rescues and
   18/19 regressions exactly, with the single miss fully explained by an input
   difference rather than a flaw in the method. The remaining 15 gate subsets can
   be trusted from the offline ablation, subject to the same ~0.4% per-claim
   retrieval instability.
