# Arm F validation — G4 disabled (G1+G2+G3)

Controlled validation of the D1 offline ablation. The adjudicator is not
redesigned: G1, G2 and G3 keep their exact semantics and evaluation order, and
G4 is still computed and recorded — it simply loses its power to veto.

> **Run status: NOT YET EXECUTED HERE.** The 504-claim run cannot be performed
> in this environment. Prerequisites checked:
>
> | requirement | status |
> |---|---|
> | GPU | **missing** — `torch 2.14.1+cpu`, `cuda.is_available() == False` |
> | `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | **not cached** |
> | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **not cached** |
> | SciFact dataset | **not cached** |
> | `all-MiniLM-L6-v2` | cached |
>
> `NLIVerifier` also defaults to `device=0`, so the pipeline cannot even
> construct here. The implementation and its tests are complete and verified;
> §5 holds the command and §6 the pre-registered expected values.

---

## 1. What changed

Smallest change that makes the gate set selectable, with the shipped rule as the
default everywhere.

| file | change |
|---|---|
| `src/prosecution.py` | `ADJUDICATION_GATE_KEYS`, `ADJUDICATION_GATE_NAMES`, `DEFAULT_ADJUDICATION_GATES`, `parse_adjudication_gates()`; `adjudicate(..., required_gates=DEFAULT_ADJUDICATION_GATES)`; each veto branch guarded by `"gN" in required`; new recorded field `adjudication_required_gates` |
| `src/pipeline.py` | `JurisVerifyPipeline(..., adjudication_gates=DEFAULT_ADJUDICATION_GATES)`, passed through to `adjudicate` |
| `eval/evaluate.py` | `--adjudication_gates`, metadata field, artifact suffix when non-default |
| `test_active_prosecution.py` | 46 new tests; fixture updated for the new attribute |
| `inspect_gate_subset_ablation.py` | reads the artifact's own rule instead of assuming all four |

**Unchanged:** retrieval, extraction, verification, prosecution candidate
selection, G1/G2/G3 semantics, LangGraph, `normalize_for_scifact_eval`. No
threshold was introduced.

### Default preservation

`adjudicate()`'s decision chain is branch-for-branch identical to the shipped
one when `required_gates` is the default; each `elif` simply gained a
`"gN" in required and` guard. The `TRUE_CONTRADICTION` reason string is also
preserved byte-for-byte under the default, so default-mode artifacts stay
comparable with every previous arm F run. A reduced selection emits a
self-describing reason instead (`passed_g1+g2+g3_section_<section>`) so a reason
string can never claim a gate was checked when it wasn't.

### Every gate outcome is still recorded

Disabling a gate removes only its veto, not its evaluation. `adjudication_gates`
continues to carry all four outcomes for every adjudicated claim — which is
exactly what made D1 reconstructable offline, and keeps the G4-off artifact
equally reconstructable.

### Rejected configuration

An empty selection is refused by `parse_adjudication_gates`. It would be arm H's
naive rule, already measured offline (71.23% overall, −62 claims), so making it
runnable would only create a way to reach that result by accident.

## 2. Tests

**267 passed** across the suite (221 before, +46 new), plus the pre-existing
unrelated `test_path` collection error in `test_orchestration_logic.py`.

| required property | tests |
|---|---|
| 1. default mode unchanged | `test_default_gate_selection_is_all_four`, `test_explicit_default_selection_matches_implicit_default`, `test_adjudicate_default_argument_preserves_shipped_behaviour`, `test_default_selection_unchanged_across_scenarios` (4 scenarios) |
| 2. G4-off still requires G1+G2+G3 | `test_g4_disabled_still_requires_g1`, `..._g2`, `..._g3`, `test_g4_disabled_admits_a_background_counter_the_default_refuses`, `test_every_gate_outcome_is_still_recorded_when_disabled` |
| 3. baseline CONTRADICT never downgraded | `test_baseline_contradict_never_downgraded_under_any_selection` (5 selections), `test_reduced_selection_never_changes_a_contradict_prediction` (2) |
| 4. prediction changes only when configured gates pass | `test_only_configured_gates_can_veto` (6 selections, asserts veto power exactly matches the selection) |
| 5. invalid configs fail clearly | `test_invalid_gate_specs_rejected` (11 bad inputs), `test_valid_gate_specs_normalised_canonically` (6), `test_empty_selection_is_rejected_as_the_naive_rule`, `test_pipeline_rejects_an_invalid_gate_selection` |
| 6. existing tests green | full suite 267 passed |

Test 1 is the strongest of these: it asserts the *entire result dict* is
identical between implicit and explicit default selection, across all four
adjudication scenarios, excluding only the new bookkeeping field.

One fixture change was required: `build_pipeline` constructs the pipeline via
`__new__` and so must now also set `adjudication_gates`. Without it, 38 existing
active-mode tests failed with `AttributeError` — caught and fixed before
proceeding.

## 3. D1 reconfirmation

`inspect_gate_subset_ablation.py` now reads the gate rule from the artifact
rather than assuming all four, and still reports **11/11 consistency checks
passed** against the default-mode 504 artifact, including the full subset
reproducing 24 / 13 / 11. When pointed at the G4-off artifact it will check
against `G1+G2+G3` automatically.

## 4. Results

| metric | arm D | arm F (G1+G2+G3+G4) | arm F-G4off (G1+G2+G3) |
|---|---|---|---|
| overall accuracy | 83.53% | 83.93% | _pending_ |
| CONTRADICT | 66.47% (115/173) | 73.99% (128/173) | _pending_ |
| SUPPORT | 92.45% (306/331) | 89.12% (295/331) | _pending_ |
| overrides | — | 24 | _pending_ |
| correct overrides | — | 13 | _pending_ |
| regressions | — | 11 | _pending_ |
| override precision | — | 54.17% | _pending_ |
| net claims | — | +2 | _pending_ |
| matches offline reconstruction | — | — | _pending_ |

## 5. Command

```bash
python eval/evaluate.py \
    --evidence_mode section --max_sentences 2 --section_weight 0.05 \
    --prosecution_mode active \
    --adjudication_gates g1,g2,g3
```

Writes `eval/results_<ts>_n504_section2_w0.05_prosactive_gatesg1g2g3.json`. The
suffix keeps it distinct from the default-mode artifact.

Then, to check it against the reconstruction and regenerate the override table:

```bash
python inspect_gate_subset_ablation.py --artifact eval/<new artifact>
python inspect_arm_f_overrides.py      --artifact eval/<new artifact> \
    --out docs/arm_f_g4off_override_analysis.md
```

## 6. Pre-registered expected values

From the D1 reconstruction, recorded **before** the run so the comparison cannot
be fitted after the fact. The run should match these **exactly** — the
reconstruction is arithmetic over stored gate outcomes, not a model
approximation, so any deviation is a real discrepancy and not noise.

| quantity | expected |
|---|---|
| candidate overrides | **43** |
| correct overrides (rescues) | **24** |
| SUPPORT regressions | **19** |
| override precision | **55.81%** |
| overall accuracy | **84.52%** (426/504) |
| overall delta vs arm D | **+0.99 pp** |
| CONTRADICT accuracy | **80.35%** (139/173) |
| SUPPORT accuracy | **86.71%** (287/331) |
| net claims vs arm D | **+5** |

Secondary expectations:

- Some overrides should now carry a `background` or `methods` counter section —
  impossible under the default rule, and the clearest single signature that G4
  is genuinely disabled. D1 predicts **4** such cases (1 background, 3 methods).
- 15 overrides should have an `unknown`-section counter from a **different**
  document than the baseline cited — the group G4 was rejecting.
- Every override's `adjudication_required_gates` should read `["g1","g2","g3"]`
  and its reason should start `passed_g1+g2+g3_section_`.

### How to read a mismatch

| observation | reading |
|---|---|
| exact match on all nine rows | D1's reconstruction method is validated; the ablation can be trusted for the remaining 15 subsets without re-running them |
| counts match, accuracy differs | arithmetic error in the reconstruction (ARM_D reference or rescoring) |
| counts differ | the counterfactual override rule in D1 does not match production's — most likely the `baseline_prediction == "SUPPORT"` ⟺ `primary_verdict != CONTRADICTED` equivalence |
| run differs *and* default-mode re-run also differs | non-determinism in the pipeline, which would invalidate every prior arm comparison and takes priority over this one |

## 7. What this validation does and does not settle

**Does settle:** whether D1's offline reconstruction predicts real runs. That
matters well beyond G4 — if it holds, the other 15 subsets need no GPU time, and
future gate questions become free.

**Does not settle:** whether G1+G2+G3 is the rule to ship. D1 already flagged
that the top six subsets span only 4 claims out of 504, so +0.99 pp versus
+0.40 pp is **inside noise** and comes from a best-of-16 selection. Confirming
the reconstruction confirms the arithmetic, not the superiority. What survives
that caution is the pattern-level evidence from D1: G3 is necessary (8/8 subsets
without it are net-negative), and G4 is the only gate whose removals are
enriched for rescues.

Also unchanged: SUPPORT accuracy is expected to fall further under G1+G2+G3
(86.71% vs 89.12%), trading 8 more regressions for 11 more rescues. Whether that
trade is acceptable is a product question this run does not answer.
