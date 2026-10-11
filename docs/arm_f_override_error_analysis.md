# Arm F override error analysis

Read-only analysis. No production code was changed to produce it.

- **Artifact:** `eval/results_20261009_085502_n504_section2_w0.05_prosactive.json`
- **Config:** evidence_mode=`section`, max_sentences=`2`, section_weight=`0.05`, prosecution_mode=`active`
- **Claims:** 504
- **Overrides found:** 24 (13 rescues, 11 regressions)

## 1. Effect on accuracy

| metric | arm D | arm F | delta |
|---|---|---|---|
| overall | 83.53% | 83.93% | +0.40 |
| CONTRADICT | 66.47% (115/173) | 73.99% (128/173) | +7.52 |
| SUPPORT | 92.45% (306/331) | 89.12% (295/331) | -3.33 |

Net: **+2 claims**. Override precision **13/24 = 54.17%**

### Did the gates discriminate, or only reduce volume?

The like-for-like baseline is NOT arm H's reported 54.86% precision: that
figure's denominator is all claims carrying counter-evidence, including
ones where an override changes nothing because arm D already said
CONTRADICT. The comparable denominator is the *overridable* population --
claims arm D predicted SUPPORT that carry counter-evidence:

- rescuable (gold CONTRADICT, baseline SUPPORT, has counter): **44** (arm H)
- losable (gold SUPPORT, baseline SUPPORT, has counter): **105** (derived from arm H's naive-rule SUPPORT of 60.73%)
- overridable population: **149**
- base rate of correctness: **29.53%**

Arm F's 54.17% is **1.83x** that base rate. A random 24 drawn from the 149 would be expected to yield 7.1 rescues and a net of -9.8 claims.

## 2. All overrides

| # | gold | base | pred | outcome | cat | counter sec | x-doc | in bundle | conf | margin | rev p(C) | prim-cnt p(C) | rank | pool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | S | S | C | **REGR** |  | unknown | no | no | 0.8638067841529846 | 0.7295141816139221 | 0.9986521601676941 | 0.002087740693241358 | 0 | 29 |
| 10 | C | S | C | rescue |  | unknown | no | yes | 0.9792534112930298 | 0.9595549497753382 | 0.9859001040458679 | 0.008848527446389198 | 0 | 26 |
| 46 | C | S | C | rescue |  | conclusions | no | no | 0.9734235405921936 | 0.9481361322104931 | 0.9885453581809998 | 0.007109354715794325 | 0 | 23 |
| 83 | C | S | C | rescue |  | unknown | no | no | 0.9817984700202942 | 0.9672288736328483 | 0.9933422207832336 | 0.0062419213354587555 | 0 | 21 |
| 93 | C | S | C | rescue |  | results | yes | no | 0.9995154142379761 | 0.9991781437711325 | 0.9932606816291809 | 0.00026056484784930944 | 0 | 21 |
| 95 | S | S | C | **REGR** |  | unknown | no | no | 0.9440388083457947 | 0.8891159817576408 | 0.9510303735733032 | 0.0009428987978026271 | 0 | 20 |
| 101 | C | S | C | rescue |  | conclusions | no | no | 0.8070629239082336 | 0.6157427132129669 | 0.9852476119995117 | 0.002118943026289344 | 0 | 24 |
| 110 | C | S | C | rescue |  | unknown | no | no | 0.9995978474617004 | 0.999364663643064 | 0.9959153532981873 | 0.0002482318377587944 | 0 | 22 |
| 122 | C | S | C | rescue |  | unknown | no | yes | 0.973422110080719 | 0.9487717300653458 | 0.9820408821105957 | 0.008848527446389198 | 0 | 22 |
| 179 | S | S | C | **REGR** |  | unknown | no | no | 0.9315764904022217 | 0.8661296218633652 | 0.9078357815742493 | 0.09362685680389404 | 0 | 8 |
| 204 | C | S | C | rescue |  | results | no | yes | 0.9925245642662048 | 0.9856415251269937 | 0.9992258548736572 | 0.003729023039340973 | 0 | 25 |
| 285 | S | S | C | **REGR** |  | unknown | no | no | 0.6332610845565796 | 0.31257492303848267 | 0.7716442346572876 | 0.003034546272829175 | 0 | 15 |
| 306 | S | S | C | **REGR** |  | unknown | no | no | 0.9928931593894958 | 0.9865684080868959 | 0.7823946475982666 | 0.0013700105482712388 | 0 | 10 |
| 324 | C | S | C | rescue |  | results | yes | yes | 0.9945830702781677 | 0.9905065111815929 | 0.9877398610115051 | 0.0007640427211299539 | 0 | 27 |
| 338 | S | S | C | **REGR** |  | unknown | no | no | 0.5853667855262756 | 0.1798950731754303 | 0.7129027843475342 | 0.008076254278421402 | 0 | 12 |
| 356 | S | S | C | **REGR** |  | unknown | no | yes | 0.8791503310203552 | 0.7651187554001808 | 0.9958035349845886 | 0.0027857704553753138 | 0 | 14 |
| 370 | C | S | C | rescue |  | unknown | no | yes | 0.9863734245300293 | 0.9754904555156827 | 0.8314900398254395 | 0.002495009219273925 | 0 | 13 |
| 403 | C | S | C | rescue |  | results | no | no | 0.8380401134490967 | 0.6770002543926239 | 0.9969639182090759 | 0.45678824186325073 | 0 | 27 |
| 412 | C | S | C | rescue |  | conclusions | no | yes | 0.9840188026428223 | 0.9685539649799466 | 0.9996213912963867 | 0.011404259130358696 | 0 | 27 |
| 418 | S | S | C | **REGR** |  | results | yes | no | 0.944595217704773 | 0.890160795301199 | 0.9049577116966248 | 0.0033103751484304667 | 0 | 27 |
| 419 | S | S | C | **REGR** |  | results | no | no | 0.9987208247184753 | 0.9976382666500285 | 0.9554209113121033 | 0.20064367353916168 | 0 | 20 |
| 420 | S | S | C | **REGR** |  | conclusions | yes | no | 0.995470404624939 | 0.9913320210762322 | 0.9960209727287292 | 0.00839473307132721 | 0 | 20 |
| 434 | C | S | C | rescue |  | unknown | no | no | 0.9691534042358398 | 0.9387662336230278 | 0.9665183424949646 | 0.0008928981842473149 | 0 | 17 |
| 452 | S | S | C | **REGR** |  | unknown | no | yes | 0.6785492897033691 | 0.3695509135723114 | 0.7429829835891724 | 0.0006917404243722558 | 0 | 17 |

`conf` = counter_confidence (argmax class probability). `margin` = counter_score (P(contradiction) minus best competitor). `rev p(C)` = P(contradiction) for claim->counter (the symmetry gate). `prim-cnt p(C)` = P(contradiction) for primary->counter (the coherence gate).

## 3. Where are the regressions concentrated?

### Counter-evidence section

| value | rescues | regressions | total | precision |
|---|---|---|---|---|
| unknown | 6 | 8 | 14 | 42.9% |
| results | 4 | 2 | 6 | 66.7% |
| conclusions | 3 | 1 | 4 | 75.0% |

### Cross-document counter-evidence

| value | rescues | regressions | total | precision |
|---|---|---|---|---|
| False | 11 | 9 | 20 | 55.0% |
| True | 2 | 2 | 4 | 50.0% |

### Counter already in arm D's bundle

| value | rescues | regressions | total | precision |
|---|---|---|---|---|
| False | 7 | 9 | 16 | 43.8% |
| True | 6 | 2 | 8 | 75.0% |

### Adjudication reason

| value | rescues | regressions | total | precision |
|---|---|---|---|---|
| symmetric_contradiction_grounded_in_unknown | 6 | 8 | 14 | 42.9% |
| symmetric_contradiction_grounded_in_results | 4 | 2 | 6 | 66.7% |
| symmetric_contradiction_grounded_in_conclusions | 3 | 1 | 4 | 75.0% |

### Reverse-direction verdict

| value | rescues | regressions | total | precision |
|---|---|---|---|---|
| CONTRADICTED | 13 | 11 | 24 | 54.2% |

### Primary-vs-counter verdict

| value | rescues | regressions | total | precision |
|---|---|---|---|---|
| PARTIALLY_SUPPORTED | 7 | 9 | 16 | 43.8% |
| SUPPORTED | 6 | 2 | 8 | 75.0% |

### Baseline verdict

| value | rescues | regressions | total | precision |
|---|---|---|---|---|
| PARTIALLY_SUPPORTED | 13 | 11 | 24 | 54.2% |

### Numeric fields: rescues vs regressions

| field | rescues (min/med/max) | regressions (min/med/max) | separation |
|---|---|---|---|
| counter_confidence | 0.8071 / 0.9818 / 0.9996 | 0.5854 / 0.9316 / 0.9987 | OVERLAPS on [0.8071, 0.9987] -- no clean split |
| counter_margin | 0.6157 / 0.9672 / 0.9994 | 0.1799 / 0.8661 / 0.9976 | OVERLAPS on [0.6157, 0.9976] -- no clean split |
| counter_p_contradiction | 0.8071 / 0.9818 / 0.9996 | 0.5854 / 0.9316 / 0.9987 | OVERLAPS on [0.8071, 0.9987] -- no clean split |
| reverse_p_contradiction | 0.8315 / 0.9885 / 0.9996 | 0.7129 / 0.9078 / 0.9987 | OVERLAPS on [0.8315, 0.9987] -- no clean split |
| primary_counter_p_contradiction | 0.0002 / 0.0037 / 0.4568 | 0.0007 / 0.003 / 0.2006 | OVERLAPS on [0.0007, 0.2006] -- no clean split |
| primary_p_contradiction | 0.0003 / 0.0413 / 0.2616 | 0.0007 / 0.0048 / 0.0292 | OVERLAPS on [0.0007, 0.0292] -- no clean split |
| max_p_contradiction | 0.8071 / 0.9818 / 0.9996 | 0.5854 / 0.9316 / 0.9987 | OVERLAPS on [0.8071, 0.9987] -- no clean split |
| counter_rank | 0 / 0 / 0 | 0 / 0 / 0 | OVERLAPS on [0.0000, 0.0000] -- no clean split |
| candidate_pool_size | 13 / 23 / 27 | 8 / 17 / 29 | OVERLAPS on [13.0000, 27.0000] -- no clean split |
| n_contradictory_candidates | 1 / 2 / 9 | 1 / 2 / 7 | OVERLAPS on [1.0000, 7.0000] -- no clean split |

Descriptive only. No threshold is proposed; overlap statements say
whether a clean split exists at all, which is the prerequisite for
asking the question later.

## 4. SUPPORT regressions (wrong overrides)

### Claim 2 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.
- **Primary evidence** (doc `44265107`, section `unknown`): Patients receiving methadone maintenance are accepted by 56% of the responding programs. Approximately 180 patients receiving methadone maintenance are reported to have undergone liver transplantation.
- **Counter-evidence** (doc `44265107`, section `unknown`, cross-doc `False`, in-bundle `False`): Ninety-four percent have addiction treatment requirements.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.8638067841529846 (conf 0.8638067841529846, margin 0.7295141816139221); claim->counter P(C)=0.9986521601676941 (CONTRADICTED); primary->counter P(C)=0.002087740693241358 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.0009942654287442565 (conf 0.9986039996147156)
- **Distributions:** counter `{'contradiction': 0.8638067841529846, 'neutral': 0.1342926025390625, 'entailment': 0.0019006739603355527}`; primary `{'neutral': 0.9986039996147156, 'contradiction': 0.0009942654287442565, 'entailment': 0.00040171953151002526}`
- **Pool:** size 29, 1 contradictory, counter at contradiction-rank 0 / pool index 8, max P(C) in pool 0.8638067841529846
- **Failure category:** **A — semantic scope mismatch**
- **Notes:** The claim measures the share of programmes requiring methadone **discontinuation**; the counter-evidence reports the share with **any addiction-treatment requirement** (94%). Both figures come from the same survey (same doc `44265107`), so they look mutually exclusive, but they quantify different attributes — a programme can have an addiction-treatment requirement without requiring discontinuation. The counter never addresses the 32% discontinuation figure.

### Claim 95 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** Commelina yellow mottle virus (ComYMV) has three typical badnavirus ORFs, capable of encoding proteins 23, 15, and 216 kD.
- **Primary evidence** (doc `14376683`, section `unknown`): We have characterized the viral transcript and determined the complete sequence of the genome of Commelina mellow mottle virus (CoYMV), a member of this group. A new badnavirus was discovered from pagoda trees showing yellow mosaic symptoms on the leaves by high throughput sequencing of small RNAs.
- **Counter-evidence** (doc `14376683`, section `unknown`, cross-doc `False`, in-bundle `False`): Analysis of the viral transcript indicates that the virus encodes a single terminally-redundant genome-length plus 120 nucleotide transcript.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9440388083457947 (conf 0.9440388083457947, margin 0.8891159817576408); claim->counter P(C)=0.9510303735733032 (CONTRADICTED); primary->counter P(C)=0.0009428987978026271 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.0007319112773984671 (conf 0.9988993406295776)
- **Distributions:** counter `{'contradiction': 0.9440388083457947, 'neutral': 0.05492282658815384, 'entailment': 0.0010383856715634465}`; primary `{'neutral': 0.9988993406295776, 'contradiction': 0.0007319112773984671, 'entailment': 0.00036874759825877845}`
- **Pool:** size 20, 2 contradictory, counter at contradiction-rank 0 / pool index 2, max P(C) in pool 0.9440388083457947
- **Failure category:** **A — semantic scope mismatch**
- **Notes:** The claim counts **three ORFs** encoding 23, 15 and 216 kD proteins; the counter reports **a single genome-length transcript**. ORFs and transcripts are different molecular objects — one transcript can carry three ORFs — so 'three' versus 'single' is not a conflict. The contradiction is an artefact of comparing counts across two different entities.

### Claim 179 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** Human embryonic stem cells give rise to cell types from all three embryonic germ layers.
- **Primary evidence** (doc `3360421`, section `unknown`): When grafted into SCID mice, both lines give rise to teratomas containing derivatives of all three embryonic germ layers. Embryonic stem cells provide a model to study early human embryology, an investigational tool for discovery of novel growth factors and medicines, and a potential source of cells for use in transplantation therapy.
- **Counter-evidence** (doc `3360421`, section `unknown`, cross-doc `False`, in-bundle `False`): Both cell lines differentiate in vitro into extraembryonic and somatic cell lineages.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9315764904022217 (conf 0.9315764904022217, margin 0.8661296218633652); claim->counter P(C)=0.9078357815742493 (CONTRADICTED); primary->counter P(C)=0.09362685680389404 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.00960313156247139 (conf 0.9824200868606567)
- **Distributions:** counter `{'contradiction': 0.9315764904022217, 'neutral': 0.0654468685388565, 'entailment': 0.0029766084626317024}`; primary `{'neutral': 0.9824200868606567, 'contradiction': 0.00960313156247139, 'entailment': 0.007976840250194073}`
- **Pool:** size 8, 1 contradictory, counter at contradiction-rank 0 / pool index 3, max P(C) in pool 0.9315764904022217
- **Failure category:** **A — semantic scope mismatch**
- **Notes:** The claim asserts derivation of **all three embryonic germ layers**, and the primary evidence supports it via teratomas in SCID mice. The counter reports differentiation **in vitro** into **extraembryonic and somatic** lineages — a different partition of cell types under a different condition. Neither the taxonomy nor the setting matches, and both statements can hold simultaneously.

### Claim 285 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** Most termination events in Okazaki fragments are dictated by initiation patterns.
- **Primary evidence** (doc `36606083`, section `unknown`): Via the deep sequencing of Okazaki fragments, we comprehensively document replication fork directionality throughout the S. Using a strain in which late origins can be induced to fire early, we show that replication termination is a largely passive phenomenon that does not rely on cis-acting sequences or replication fork pausing.
- **Counter-evidence** (doc `36606083`, section `unknown`, cross-doc `False`, in-bundle `False`): The replication profile is predominantly determined by the kinetics of origin firing, allowing us to reconstruct chromosome-wide timing profiles from an asynchronous culture.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.6332610845565796 (conf 0.6332610845565796, margin 0.31257492303848267); claim->counter P(C)=0.7716442346572876 (CONTRADICTED); primary->counter P(C)=0.003034546272829175 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.020328883081674576 (conf 0.9605551958084106)
- **Distributions:** counter `{'contradiction': 0.6332610845565796, 'neutral': 0.3206861615180969, 'entailment': 0.04605282098054886}`; primary `{'neutral': 0.9605551958084106, 'contradiction': 0.020328883081674576, 'entailment': 0.019115878269076347}`
- **Pool:** size 15, 2 contradictory, counter at contradiction-rank 0 / pool index 5, max P(C) in pool 0.6332610845565796
- **Failure category:** **C — NLI error**
- **Notes:** Scope and relevance are correct: same document, same phenomenon, same proposition. The claim says termination is dictated by **initiation patterns**; the counter says the replication profile is determined by the **kinetics of origin firing** — and origin firing *is* initiation. The two statements agree, so the contradiction verdict is an NLI error on paraphrased causal vocabulary. Consistent with the weakest signal in the set: counter P(C)=0.633, margin 0.313, the lowest of the 11.

### Claim 306 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** Notch signaling occurs between tumor cells and stromal cells.
- **Primary evidence** (doc `16361581`, section `unknown`): We recently demonstrated that Notch signaling is involved in proliferation and survival of B cell-derived tumor cells of classic Hodgkin disease and described a novel mechanism for the oncogenic capacity of Notch. In this study we investigated whether Notch signaling is involved in the tight interactions between neoplastic plasma cells and their bone marrow microenvironment, which are essential f…
- **Counter-evidence** (doc `16361581`, section `unknown`, cross-doc `False`, in-bundle `False`): Notch receptors expressed on hematopoietic stem cells interact with their ligands on bone marrow stromal cells and thereby control cell fate decisions and survival.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9928931593894958 (conf 0.9928931593894958, margin 0.9865684080868959); claim->counter P(C)=0.7823946475982666 (CONTRADICTED); primary->counter P(C)=0.0013700105482712388 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.005353700835257769 (conf 0.647867739200592)
- **Distributions:** counter `{'contradiction': 0.9928931593894958, 'neutral': 0.006324751302599907, 'entailment': 0.0007820306927897036}`; primary `{'neutral': 0.647867739200592, 'entailment': 0.3467785716056824, 'contradiction': 0.005353700835257769}`
- **Pool:** size 10, 1 contradictory, counter at contradiction-rank 0 / pool index 0, max P(C) in pool 0.9928931593894958
- **Failure category:** **A — semantic scope mismatch**
- **Notes:** Both sentences concern Notch signalling with bone-marrow stromal cells, but the claim's participants are **tumour cells** and the counter's are **haematopoietic stem cells**. The relation asserted is the same and neither statement excludes the other; only the population differs.

### Claim 338 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** Pioneer factor OCT3/4 interacts with major chromatin remodeling factors.
- **Primary evidence** (doc `11011905`, section `unknown`): Purified Oct4, Sox2, and Klf4 proteins can bind nucleosomes in vitro, and in vivo they preferentially target silent sites enriched for nucleosomes. Our findings provide insight into how pioneer factors can target naive chromatin sites.
- **Counter-evidence** (doc `11011905`, section `unknown`, cross-doc `False`, in-bundle `False`): FoxA, the paradigm pioneer TF, has a winged helix DBD that resembles linker histone and thereby binds its target sites on nucleosomes and in compacted chromatin.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.5853667855262756 (conf 0.5853667855262756, margin 0.1798950731754303); claim->counter P(C)=0.7129027843475342 (CONTRADICTED); primary->counter P(C)=0.008076254278421402 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.004798168782144785 (conf 0.991569459438324)
- **Distributions:** counter `{'contradiction': 0.5853667855262756, 'neutral': 0.40547171235084534, 'entailment': 0.009161518886685371}`; primary `{'neutral': 0.991569459438324, 'contradiction': 0.004798168782144785, 'entailment': 0.0036324739921838045}`
- **Pool:** size 12, 1 contradictory, counter at contradiction-rank 0 / pool index 0, max P(C) in pool 0.5853667855262756
- **Failure category:** **A — semantic scope mismatch**
- **Notes:** The claim is about **OCT3/4**; the counter describes **FoxA**, a different pioneer transcription factor, and its winged-helix DBD. The counter says nothing about OCT3/4's interactions, so it cannot refute the claim. The shared frame ('pioneer factor', nucleosome binding) is what the NLI model reacted to — counter P(C)=0.585, margin 0.180, the weakest confidence of the 11.

### Claim 356 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** Pyridostatin reduces proliferation of homologous recombination - defective cells.
- **Primary evidence** (doc `16472469`, section `unknown`): Treatment with the G4-stabilizing compound pyridostatin (PDS) increases telomere fragility in BRCA2-deficient cells, suggesting that G4 formation drives telomere instability. BLM, WRN, and p53 are involved in the homologous DNA recombination pathway.
- **Counter-evidence** (doc `16472469`, section `unknown`, cross-doc `False`, in-bundle `True`): Treatment with the G4-stabilizing compound pyridostatin (PDS) increases telomere fragility in BRCA2-deficient cells, suggesting that G4 formation drives telomere instability.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.8791503310203552 (conf 0.8791503310203552, margin 0.7651187554001808); claim->counter P(C)=0.9958035349845886 (CONTRADICTED); primary->counter P(C)=0.0027857704553753138 (SUPPORTED); primary->claim P(C)=0.020641831681132317 (conf 0.9782038927078247)
- **Distributions:** counter `{'contradiction': 0.8791503310203552, 'neutral': 0.11403157562017441, 'entailment': 0.006818100344389677}`; primary `{'neutral': 0.9782038927078247, 'contradiction': 0.020641831681132317, 'entailment': 0.0011543420841917396}`
- **Pool:** size 14, 3 contradictory, counter at contradiction-rank 0 / pool index 3, max P(C) in pool 0.8791503310203552
- **Failure category:** **D — provenance/grounding failure**
- **Notes:** The counter-evidence is **verbatim the first sentence of the primary evidence** (in-bundle `True`, and primary→counter is SUPPORTED at P(C)=0.0028 — the premise trivially entails its own substring). It is therefore not independent counter-evidence at all: the apparent contradiction rests on comparing the claim against a fragment of the premise that produced its own baseline verdict, and the pool-coherence gate passed vacuously by self-comparison. Secondarily the endpoint also differs — the claim is about **proliferation**, the counter about **telomere fragility**.

### Claim 418 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** The DESMOND program demonstrates no significant impact on biochemical outcomes.
- **Primary evidence** (doc `29124963`, section `results`): The overall evidence rating was low to very low for important outcomes such as pre-eclampsia, gestational diabetes, gestational hypertension, and preterm delivery. No patients received ALK inhibitor therapy.
- **Counter-evidence** (doc `11360768`, section `results`, cross-doc `True`, in-bundle `False`): Dietary intervention resulted in the largest reduction in maternal gestational weight gain (3.84 kg, 2.45 to 5.22 kg), with improved pregnancy outcomes compared with other interventions.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_results`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.944595217704773 (conf 0.944595217704773, margin 0.890160795301199); claim->counter P(C)=0.9049577116966248 (CONTRADICTED); primary->counter P(C)=0.0033103751484304667 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.001977876527234912 (conf 0.9911051392555237)
- **Distributions:** counter `{'contradiction': 0.944595217704773, 'neutral': 0.05443442240357399, 'entailment': 0.0009703457471914589}`; primary `{'neutral': 0.9911051392555237, 'entailment': 0.0069169639609754086, 'contradiction': 0.001977876527234912}`
- **Pool:** size 27, 3 contradictory, counter at contradiction-rank 0 / pool index 8, max P(C) in pool 0.944595217704773
- **Failure category:** **B — unrelated evidence**
- **Notes:** The claim concerns the **DESMOND** programme and biochemical outcomes. The counter (cross-doc `11360768`) reports **dietary intervention** reducing **maternal gestational weight gain** in pregnancy. The primary evidence is equally off-subject (pre-eclampsia, gestational diabetes, 'No patients received ALK inhibitor therapy'). Neither sentence addresses DESMOND, so retrieval failed before adjudication ran; the counter is topically adjacent ('intervention improves outcomes') without addressing the proposition.

### Claim 419 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** The DESMOND program demonstrates no significant impact on lifestyles outcomes.
- **Primary evidence** (doc `11360768`, section `results`): With all interventions combined, there were no significant differences in birth weight (mean difference -50 g, -100 to 0 g) and the incidence of large for gestational age (relative risk 0.85, 0.66 to 1.09) or small for gestational age (1.00, 0.78 to 1.28) babies between the groups, though by itself physical activity was associated with reduced birth weight (mean difference -60 g, -120 to -10 g). …
- **Counter-evidence** (doc `11360768`, section `results`, cross-doc `False`, in-bundle `False`): Dietary intervention resulted in the largest reduction in maternal gestational weight gain (3.84 kg, 2.45 to 5.22 kg), with improved pregnancy outcomes compared with other interventions.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_results`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9987208247184753 (conf 0.9987208247184753, margin 0.9976382666500285); claim->counter P(C)=0.9554209113121033 (CONTRADICTED); primary->counter P(C)=0.20064367353916168 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.0014786930987611413 (conf 0.997079610824585)
- **Distributions:** counter `{'contradiction': 0.9987208247184753, 'neutral': 0.001082558068446815, 'entailment': 0.00019657173834275454}`; primary `{'neutral': 0.997079610824585, 'contradiction': 0.0014786930987611413, 'entailment': 0.0014416207559406757}`
- **Pool:** size 20, 7 contradictory, counter at contradiction-rank 0 / pool index 8, max P(C) in pool 0.9987208247184753
- **Failure category:** **B — unrelated evidence**
- **Notes:** Same failure as claim 418 for the lifestyle-outcomes variant. Both primary and counter come from doc `11360768`, a study of pregnancy interventions and gestational weight gain — not of **DESMOND**. The retrieved document is simply the wrong subject for the claim, so the counter cannot bear on it.

### Claim 420 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** The DESMOND program demonstrates no significant impact on weight loss.
- **Primary evidence** (doc `40949706`, section `unknown`): The groups differed on weight lost at 24 months. Surgery generates substantial weight loss, but 20–30% fails to achieve successful weight loss.
- **Counter-evidence** (doc `6841927`, section `conclusions`, cross-doc `True`, in-bundle `False`): CONCLUSIONS This study demonstrates the ability of TTM-based tailored feedback to improve healthy eating, exercise, managing emotional distress, and weight on a population basis.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_conclusions`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.995470404624939 (conf 0.995470404624939, margin 0.9913320210762322); claim->counter P(C)=0.9960209727287292 (CONTRADICTED); primary->counter P(C)=0.00839473307132721 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.029166357591748238 (conf 0.9702590107917786)
- **Distributions:** counter `{'contradiction': 0.995470404624939, 'neutral': 0.00413838354870677, 'entailment': 0.0003911823150701821}`; primary `{'neutral': 0.9702590107917786, 'contradiction': 0.029166357591748238, 'entailment': 0.0005746020469814539}`
- **Pool:** size 20, 3 contradictory, counter at contradiction-rank 0 / pool index 8, max P(C) in pool 0.995470404624939
- **Failure category:** **B — unrelated evidence**
- **Notes:** The claim is about **DESMOND** and weight loss; the counter (cross-doc `6841927`) reports **TTM-based tailored feedback** improving healthy eating, exercise and weight. Weight is the only shared term — the programmes are different, so a success for TTM-based feedback says nothing about DESMOND. The primary evidence is likewise unrelated (bariatric surgery weight loss).

### Claim 452 (gold SUPPORT, SUPPORT -> CONTRADICT)

- **Claim:** The recruitment of Wdr5 to its target loci depends on Kat8.
- **Primary evidence** (doc `11615422`, section `unknown`): FISH showed that, although they were heterogeneous, breakpoints on 9p resulted in the partial or complete deletion of PAX5. This approach identified 5 fusion gene partners with PAX5: LOC392027 (7p12.1), SLCO1B3 (12p12), ASXL1 (20q11.1), KIF3B (20q11.21), and C20orf112 (20q11.1).
- **Counter-evidence** (doc `11615422`, section `unknown`, cross-doc `False`, in-bundle `True`): FISH showed that, although they were heterogeneous, breakpoints on 9p resulted in the partial or complete deletion of PAX5.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.6785492897033691 (conf 0.6785492897033691, margin 0.3695509135723114); claim->counter P(C)=0.7429829835891724 (CONTRADICTED); primary->counter P(C)=0.0006917404243722558 (SUPPORTED); primary->claim P(C)=0.0038709603250026703 (conf 0.995561957359314)
- **Distributions:** counter `{'contradiction': 0.6785492897033691, 'neutral': 0.30899837613105774, 'entailment': 0.01245236024260521}`; primary `{'neutral': 0.995561957359314, 'contradiction': 0.0038709603250026703, 'entailment': 0.000567000126466155}`
- **Pool:** size 17, 1 contradictory, counter at contradiction-rank 0 / pool index 2, max P(C) in pool 0.6785492897033691
- **Failure category:** **D — provenance/grounding failure**
- **Notes:** The counter-evidence is **verbatim the first sentence of the primary evidence** (in-bundle `True`, primary→counter SUPPORTED at P(C)=0.0007), so the coherence gate again passed by self-comparison rather than on independent evidence. Compounding it, the retrieved document is on an entirely different subject: the claim concerns **Wdr5 recruitment depending on Kat8**, while both premise and counter describe **PAX5 deletion from 9p breakpoints detected by FISH**. No proposition-level test would be needed to reject this; retrieval should not have produced it.

### Failure-category distribution (11 SUPPORT regressions)

| Failure category | Count | Percentage |
|---|---:|---:|
| A — semantic scope mismatch | 5 | 45.5% |
| B — unrelated evidence | 3 | 27.3% |
| C — NLI error | 1 | 9.1% |
| D — provenance/grounding failure | 2 | 18.2% |
| E — other | 0 | 0.0% |
| **total** | **11** | **100.0%** |

Claim ids by category — A: 2, 95, 179, 306, 338 · B: 418, 419, 420 · C: 285 · D: 356, 452 · E: none.

**Dominant pattern.** 5 of the 11 regressions (45.5%) are
**semantic scope mismatch**: the counter-evidence shares the claim's topic and
framing but diverges on one specific dimension — a different measured attribute
(claim 2), a different molecular entity (95), a different taxonomy and
experimental condition (179), a different cell population (306), or a different
transcription factor (338). In every one of these the two statements can hold
simultaneously, so no genuine contradiction exists to detect. A further
3 (27.3%) are **unrelated evidence** — all three DESMOND claims
(418, 419, 420), where retrieval returned documents about pregnancy or
weight-management interventions that never mention the programme the claim names,
so adjudication was operating on the wrong subject entirely. Taken together,
8 of 11 (72.7%) fail because the counter-evidence does not
address the claim's proposition, not because the NLI decision on a well-matched
pair was wrong — only claim 285 (9.1%) is a genuine NLI error, and it is also the
weakest signal in the set (counter P(C)=0.633, margin 0.313).

The remaining 2 (18.2%) are a distinct and mechanically simpler defect:
in claims 356 and 452 the counter-evidence is a **verbatim substring of the
primary evidence** (the only two regressions with in-bundle `True`), so the
pool-coherence gate passed by comparing the premise against itself —
primary→counter is SUPPORTED at P(C)=0.0028 and 0.0007 respectively — and the
"counter-evidence" was never independent. This is the one failure mode here that
a non-semantic check would catch outright, since it needs no judgement about
scope: a counter-evidence sentence contained in the primary premise cannot
contradict the claim that premise was selected to support. It is recorded as a
qualitative observation; it does not alter any quantitative result in §1-§3.

### Qualitative observation — an evidence-integrity safeguard (not evaluated)

Separately from the semantic failure modes above, the audit surfaced an
**evidence-integrity** problem that none of the adjudication gates is designed to
catch. In **2 of the 11 SUPPORT regressions (18.2%)** — claims **356** and
**452** — the selected counter-evidence is a **verbatim substring of the primary
evidence**. The two sentences are therefore not independent pieces of evidence,
and the contradiction decision rests on comparing the claim against part of the
premise that was selected to support it.

This is a different kind of defect from a scope mismatch (A), unrelated evidence
(B), or a mistaken NLI call (C): the gates G1–G3 reason about the *semantic
relation* between claim, premise and counter-evidence, and all three can pass
perfectly well on a pair that is not independent in the first place. It suggests
a **deterministic evidence-independence / containment check** — verifying that
the counter-evidence is not contained in the primary premise — as a separate
safeguard alongside the semantic gates rather than as another semantic gate.

Two limits on how far this should be read. The check is **not validated**: its
quantitative effect has not been measured, and nothing here establishes that
adding it would improve overall accuracy, since rejecting an override changes
the prediction only for the claims it touches. And the 18.2% figure is a share
**of these 11 regressions**, not of the 504-claim population or of all overrides.
It is recorded as a qualitative observation and a candidate safeguard for future
work, not as a result.

## 5. Rescued CONTRADICT claims (correct overrides)

### Claim 10 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** A diminished ovarian reserve is a very strong indicator of infertility, even in an a priori non-infertile population.
- **Primary evidence** (doc `13497630`, section `unknown`): Objective To determine the associations between biomarkers of ovarian reserve and reproductive potential among women of late reproductive age. Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with …
- **Counter-evidence** (doc `13497630`, section `unknown`, cross-doc `False`, in-bundle `True`): Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9792534112930298 (conf 0.9792534112930298, margin 0.9595549497753382); claim->counter P(C)=0.9859001040458679 (CONTRADICTED); primary->counter P(C)=0.008848527446389198 (SUPPORTED); primary->claim P(C)=0.08191905170679092 (conf 0.9171388745307922)
- **Distributions:** counter `{'contradiction': 0.9792534112930298, 'neutral': 0.019698461517691612, 'entailment': 0.0010480661876499653}`; primary `{'neutral': 0.9171388745307922, 'contradiction': 0.08191905170679092, 'entailment': 0.0009421409340575337}`
- **Pool:** size 26, 1 contradictory, counter at contradiction-rank 0 / pool index 10, max P(C) in pool 0.9792534112930298
- **Why genuinely relevant:** Same population the claim singles out — women aged 30–44 **without a history of infertility** — same exposure (diminished-ovarian-reserve biomarkers) and same outcome (fertility). The cited conclusion states those biomarkers were **not associated with reduced fertility**, which directly negates 'a very strong indicator of infertility' on the claim's own terms.

### Claim 46 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Asymptomatic visual impairment screening in elderly populations leads to improved vision.
- **Primary evidence** (doc `26016929`, section `background`): OBJECTIVE To assess whether population screening for impaired vision among older people in the community leads to improvements in vision. Visual impairment in this age group can usually be reduced with treatment.
- **Counter-evidence** (doc `26016929`, section `conclusions`, cross-doc `False`, in-bundle `False`): CONCLUSIONS Screening of asymptomatic older people in the community is not justified on present evidence.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_conclusions`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9734235405921936 (conf 0.9734235405921936, margin 0.9481361322104931); claim->counter P(C)=0.9885453581809998 (CONTRADICTED); primary->counter P(C)=0.007109354715794325 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.002031253883615136 (conf 0.996217668056488)
- **Distributions:** counter `{'contradiction': 0.9734235405921936, 'neutral': 0.025287408381700516, 'entailment': 0.0012889993377029896}`; primary `{'neutral': 0.996217668056488, 'contradiction': 0.002031253883615136, 'entailment': 0.00175105023663491}`
- **Pool:** size 23, 2 contradictory, counter at contradiction-rank 0 / pool index 9, max P(C) in pool 0.9734235405921936
- **Why genuinely relevant:** Population (asymptomatic older people), intervention (screening) and outcome (benefit) all match, and the direction inverts: the claim says screening **leads to improved vision**, the conclusion says it is **not justified on present evidence**.

### Claim 83 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Charcoal is an effective treatment for acute paraquat poisoning.
- **Primary evidence** (doc `3430789`, section `unknown`): The present study retrospectively analyzed 19 patients diagnosed with paraquat (PQ) poisoning with the aim to investigate the effect of activated charcoal hemoperfusion on renal function and PQ elimination. There was a linear correlation between plasma and urine paraquat concentration prior to and during activated charcoal hemoperfusion.
- **Counter-evidence** (doc `3430789`, section `unknown`, cross-doc `False`, in-bundle `False`): Activated charcoal hemoperfusion had little effect on avoiding acute kidney injury in patients with severe PQ poisoning.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9817984700202942 (conf 0.9817984700202942, margin 0.9672288736328483); claim->counter P(C)=0.9933422207832336 (CONTRADICTED); primary->counter P(C)=0.0062419213354587555 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.00025603760150261223 (conf 0.9993451237678528)
- **Distributions:** counter `{'contradiction': 0.9817984700202942, 'neutral': 0.014569596387445927, 'entailment': 0.0036319755017757416}`; primary `{'neutral': 0.9993451237678528, 'entailment': 0.00039888915489427745, 'contradiction': 0.00025603760150261223}`
- **Pool:** size 21, 1 contradictory, counter at contradiction-rank 0 / pool index 20, max P(C) in pool 0.9817984700202942
- **Why genuinely relevant:** Same intervention (activated charcoal) and same population (severe paraquat poisoning). The counter reports it **had little effect**, inverting the claim's 'effective treatment'. The endpoint narrows from general effectiveness to acute kidney injury, but the direction still reverses on a shared axis.

### Claim 93 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Cognitive behavioral therapy is an ineffective treatment for insomnia
- **Primary evidence** (doc `24346598`, section `conclusions`): OBJECTIVE To examine short- and long-term clinical efficacy of cognitive behavioral therapy (CBT) and pharmacological treatment in older adults experiencing chronic primary insomnia. Data from this study on the treatment of chronic, severe insomnia (a small subset of all insomnia) are not generalizable to the typical insomnia patient.
- **Counter-evidence** (doc `5912283`, section `results`, cross-doc `True`, in-bundle `False`): RESULTS CBT resulted in improved short- and long-term outcomes compared with zopiclone on 3 out of 4 outcome measures.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_results`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9995154142379761 (conf 0.9995154142379761, margin 0.9991781437711325); claim->counter P(C)=0.9932606816291809 (CONTRADICTED); primary->counter P(C)=0.00026056484784930944 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.007475535850971937 (conf 0.9920955300331116)
- **Distributions:** counter `{'contradiction': 0.9995154142379761, 'neutral': 0.00033727046684361994, 'entailment': 0.00014734255091752857}`; primary `{'neutral': 0.9920955300331116, 'contradiction': 0.007475535850971937, 'entailment': 0.0004290103097446263}`
- **Pool:** size 21, 5 contradictory, counter at contradiction-rank 0 / pool index 7, max P(C) in pool 0.9995154142379761
- **Why genuinely relevant:** Identical intervention (CBT), population (insomnia) and outcome (efficacy). The claim calls CBT **ineffective**; the Results sentence reports **improved short- and long-term outcomes** on 3 of 4 measures.

### Claim 101 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Cost effectiveness evaluations based on cRCT data accurately reflect prices for patients in actual clinical practice.
- **Primary evidence** (doc `15048300`, section `unknown`): The objective of this study was to evaluate the external validity of published cost-effectiveness studies by comparing the data used in these studies (typically based on RCTs) to observational data from actual clinical practice. CONCLUSION Evaluating the cost effectiveness of prognostic biomarkers is important even when effects at an individual level are small.
- **Counter-evidence** (doc `15048300`, section `conclusions`, cross-doc `False`, in-bundle `False`): CONCLUSIONS The published cost-effectiveness analyses of coxibs lacked external validity, did not represent patients in actual clinical practice, and should not have been used to inform prescribing policies.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_conclusions`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.8070629239082336 (conf 0.8070629239082336, margin 0.6157427132129669); claim->counter P(C)=0.9852476119995117 (CONTRADICTED); primary->counter P(C)=0.002118943026289344 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.011806623078882694 (conf 0.9867749214172363)
- **Distributions:** counter `{'contradiction': 0.8070629239082336, 'neutral': 0.19132021069526672, 'entailment': 0.0016169004375115037}`; primary `{'neutral': 0.9867749214172363, 'contradiction': 0.011806623078882694, 'entailment': 0.001418386586010456}`
- **Pool:** size 24, 1 contradictory, counter at contradiction-rank 0 / pool index 11, max P(C) in pool 0.8070629239082336
- **Why genuinely relevant:** Same object (cost-effectiveness analyses built on trial data) and same outcome (external validity). The claim says they **accurately reflect** actual clinical practice; the conclusion says they **lacked external validity, did not represent patients in actual clinical practice**.

### Claim 110 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Deltex has no known interactions with eIF3f.
- **Primary evidence** (doc `9505448`, section `unknown`): We show that eIF3f has an intrinsic DUB activity. Knocking down eIF3f leads to an accumulation of monoubiquitinated forms of activated Notch, an effect counteracted by murine WT eIF3f but not by a catalytically inactive mutant.
- **Counter-evidence** (doc `9505448`, section `unknown`, cross-doc `False`, in-bundle `False`): We also show that eIF3f is recruited to activated Notch on endocytic vesicles by the putative E3 ubiquitin ligase Deltex1, which serves as a bridging factor.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9995978474617004 (conf 0.9995978474617004, margin 0.999364663643064); claim->counter P(C)=0.9959153532981873 (CONTRADICTED); primary->counter P(C)=0.0002482318377587944 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.004980333615094423 (conf 0.9946310520172119)
- **Distributions:** counter `{'contradiction': 0.9995978474617004, 'neutral': 0.0002331838186364621, 'entailment': 0.00016894302098080516}`; primary `{'neutral': 0.9946310520172119, 'contradiction': 0.004980333615094423, 'entailment': 0.0003885690530296415}`
- **Pool:** size 22, 9 contradictory, counter at contradiction-rank 0 / pool index 5, max P(C) in pool 0.9995978474617004
- **Why genuinely relevant:** The same two entities and the same relation. The claim asserts Deltex has **no known interactions** with eIF3f; the counter describes eIF3f being recruited to Notch **by Deltex1 acting as a bridging factor** — a direct polarity reversal, not a scope shift.

### Claim 122 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Diminished ovarian reserve is a reliable indicator of infertility in a non-infertile population.
- **Primary evidence** (doc `13497630`, section `unknown`): Objective To determine the associations between biomarkers of ovarian reserve and reproductive potential among women of late reproductive age. Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with …
- **Counter-evidence** (doc `13497630`, section `unknown`, cross-doc `False`, in-bundle `True`): Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.973422110080719 (conf 0.973422110080719, margin 0.9487717300653458); claim->counter P(C)=0.9820408821105957 (CONTRADICTED); primary->counter P(C)=0.008848527446389198 (SUPPORTED); primary->claim P(C)=0.18285568058490753 (conf 0.8155105113983154)
- **Distributions:** counter `{'contradiction': 0.973422110080719, 'neutral': 0.02465038001537323, 'entailment': 0.001927470206283033}`; primary `{'neutral': 0.8155105113983154, 'contradiction': 0.18285568058490753, 'entailment': 0.0016338246641680598}`
- **Pool:** size 22, 4 contradictory, counter at contradiction-rank 0 / pool index 10, max P(C) in pool 0.973422110080719
- **Why genuinely relevant:** Same structure as claim 10: matching population (no prior infertility), exposure (ovarian-reserve biomarkers) and outcome, with the conclusion reporting **no association with reduced fertility** against the claim's 'reliable indicator of infertility'.

### Claim 204 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Incidence of heart failure increased by 10% in women since 1979.
- **Primary evidence** (doc `19675911`, section `conclusions`): RESULTS The incidence of heart failure was higher among men (378/100 000 persons; 95% confidence interval [CI], 361-395 for men; 289/100 000 persons; 95% CI, 277-300 for women) and did not change over time among men or women. CONCLUSION In this community-based cohort, the incidence of heart failure has not declined during 2 decades, but survival after onset of heart failure has increased overall,…
- **Counter-evidence** (doc `19675911`, section `results`, cross-doc `False`, in-bundle `True`): RESULTS The incidence of heart failure was higher among men (378/100 000 persons; 95% confidence interval [CI], 361-395 for men; 289/100 000 persons; 95% CI, 277-300 for women) and did not change over time among men or women.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_results`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9925245642662048 (conf 0.9925245642662048, margin 0.9856415251269937); claim->counter P(C)=0.9992258548736572 (CONTRADICTED); primary->counter P(C)=0.003729023039340973 (SUPPORTED); primary->claim P(C)=0.2615734040737152 (conf 0.736471951007843)
- **Distributions:** counter `{'contradiction': 0.9925245642662048, 'neutral': 0.006883039139211178, 'entailment': 0.0005923981079831719}`; primary `{'neutral': 0.736471951007843, 'contradiction': 0.2615734040737152, 'entailment': 0.0019546877592802048}`
- **Pool:** size 25, 1 contradictory, counter at contradiction-rank 0 / pool index 5, max P(C) in pool 0.9925245642662048
- **Why genuinely relevant:** Population (women), outcome (heart-failure incidence) and temporal frame (change over time) all match. The claim says incidence **increased by 10% since 1979**; the Results sentence states it **did not change over time among men or women**.

### Claim 324 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Patients with common epithelial cancers are less likely to have an emergency event as their first hospital admission if they live in resource-deprived areas.
- **Primary evidence** (doc `5641851`, section `results`): RESULTS Patients from deprived areas, older people, and women were more likely to be admitted as emergencies. Patients with colonic cancer were very significantly less likely to receive any treatment if they came from any but the most affluent area (ORs 0.639, 0.603 and 0.544 in increasingly deprived quartiles), this may have been exacerbated if the hospital was distant from their residence (OR f…
- **Counter-evidence** (doc `16390264`, section `results`, cross-doc `True`, in-bundle `True`): RESULTS Patients from deprived areas, older people, and women were more likely to be admitted as emergencies.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_results`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9945830702781677 (conf 0.9945830702781677, margin 0.9905065111815929); claim->counter P(C)=0.9877398610115051 (CONTRADICTED); primary->counter P(C)=0.0007640427211299539 (SUPPORTED); primary->claim P(C)=0.26113519072532654 (conf 0.7312997579574585)
- **Distributions:** counter `{'contradiction': 0.9945830702781677, 'neutral': 0.004076559096574783, 'entailment': 0.0013403738848865032}`; primary `{'neutral': 0.7312997579574585, 'contradiction': 0.26113519072532654, 'entailment': 0.007565015461295843}`
- **Pool:** size 27, 3 contradictory, counter at contradiction-rank 0 / pool index 4, max P(C) in pool 0.9945830702781677
- **Why genuinely relevant:** Same population (patients from deprived areas) and same outcome (emergency admission). The claim says such patients are **less likely** to present as an emergency; the Results sentence says they were **more likely to be admitted as emergencies** — a clean directional inversion.

### Claim 370 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Removal of H3K9me3 by ectopic expression of other H3K9 demethylases decreases reprogramming efficiency in SCNT experiments.
- **Primary evidence** (doc `15803282`, section `unknown`): Here we show that overexpression of a related H3K9me3 demethylase KDM4A improves human SCNT, and that, as in mice, H3K9me3 in the human somatic cell genome is an SCNT reprogramming barrier. Furthermore, we show that interactions between RNA and KDM4E are essential for H3K9 demethylation during EGA.
- **Counter-evidence** (doc `15803282`, section `unknown`, cross-doc `False`, in-bundle `True`): Here we show that overexpression of a related H3K9me3 demethylase KDM4A improves human SCNT, and that, as in mice, H3K9me3 in the human somatic cell genome is an SCNT reprogramming barrier.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9863734245300293 (conf 0.9863734245300293, margin 0.9754904555156827); claim->counter P(C)=0.8314900398254395 (CONTRADICTED); primary->counter P(C)=0.002495009219273925 (SUPPORTED); primary->claim P(C)=0.11177965253591537 (conf 0.882332980632782)
- **Distributions:** counter `{'contradiction': 0.9863734245300293, 'neutral': 0.0108829690143466, 'entailment': 0.002743595512583852}`; primary `{'neutral': 0.882332980632782, 'contradiction': 0.11177965253591537, 'entailment': 0.0058873374946415424}`
- **Pool:** size 13, 1 contradictory, counter at contradiction-rank 0 / pool index 3, max P(C) in pool 0.9863734245300293
- **Why genuinely relevant:** Same intervention class (ectopic H3K9me3 demethylase expression) and same outcome (SCNT reprogramming efficiency). The claim says removal **decreases** efficiency; the counter shows KDM4A overexpression **improves** human SCNT and identifies H3K9me3 as a reprogramming **barrier**.

### Claim 403 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Systemic immunosuppressive (IS) therapy increases the chance of cancer mortality in patients with inflammatory eye disease (IED)
- **Primary evidence** (doc `21009874`, section `methods`): DESIGN Retrospective cohort study evaluating overall and cancer mortality in relation to immunosuppressive drug exposure among patients with ocular inflammatory diseases. For patients unexposed to immunosuppressive treatment, risks of death overall (standardised mortality ratio 1.02, 95% confidence interval [CI] 0.94 to 1.11) and from cancer (1.10, 0.93 to 1.29) were similar to those of the US po…
- **Counter-evidence** (doc `21009874`, section `results`, cross-doc `False`, in-bundle `False`): Patients who used azathioprine, methotrexate, mycophenolate mofetil, ciclosporin, systemic corticosteroids, or dapsone had overall and cancer mortality similar to that of patients who never took immunosuppressive drugs.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_results`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.8380401134490967 (conf 0.8380401134490967, margin 0.6770002543926239); claim->counter P(C)=0.9969639182090759 (CONTRADICTED); primary->counter P(C)=0.45678824186325073 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.013552607968449593 (conf 0.9716930985450745)
- **Distributions:** counter `{'contradiction': 0.8380401134490967, 'neutral': 0.16103985905647278, 'entailment': 0.0009199948399327695}`; primary `{'neutral': 0.9716930985450745, 'entailment': 0.014754380099475384, 'contradiction': 0.013552607968449593}`
- **Pool:** size 27, 2 contradictory, counter at contradiction-rank 0 / pool index 12, max P(C) in pool 0.8380401134490967
- **Why genuinely relevant:** Same intervention (systemic immunosuppressives, named drug by drug), same population (inflammatory eye disease) and same endpoint (cancer mortality). The claim says IS therapy **increases** cancer mortality; the counter reports mortality **similar to that of patients who never took** these drugs.

### Claim 412 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** Taking 400mg of α-tocopheryl acetate in combination with vitamin C reduces the risk of prostate cancer.
- **Primary evidence** (doc `12009265`, section `conclusions`): CONCLUSIONS In this large, long-term trial of male physicians, neither vitamin E nor C supplementation reduced the risk of prostate or total cancer. Some human and animal studies indicate that plasma concentrations of gamma-tocopherol are inversely associated with the incidence of cardiovascular disease and prostate cancer.
- **Counter-evidence** (doc `12009265`, section `conclusions`, cross-doc `False`, in-bundle `True`): CONCLUSIONS In this large, long-term trial of male physicians, neither vitamin E nor C supplementation reduced the risk of prostate or total cancer.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_conclusions`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9840188026428223 (conf 0.9840188026428223, margin 0.9685539649799466); claim->counter P(C)=0.9996213912963867 (CONTRADICTED); primary->counter P(C)=0.011404259130358696 (SUPPORTED); primary->claim P(C)=0.20073574781417847 (conf 0.7962546348571777)
- **Distributions:** counter `{'contradiction': 0.9840188026428223, 'neutral': 0.015464837662875652, 'entailment': 0.0005162886227481067}`; primary `{'neutral': 0.7962546348571777, 'contradiction': 0.20073574781417847, 'entailment': 0.003009560750797391}`
- **Pool:** size 27, 2 contradictory, counter at contradiction-rank 0 / pool index 14, max P(C) in pool 0.9840188026428223
- **Why genuinely relevant:** Same intervention (vitamin E plus vitamin C), same population and same endpoint (prostate cancer risk). The claim says the combination **reduces** risk; the trial conclusion states **neither vitamin E nor C supplementation reduced** the risk of prostate or total cancer.

### Claim 434 (gold CONTRADICT, SUPPORT -> CONTRADICT)

- **Claim:** The density of cytokine receptor bearing cells has no effect on the distance over which cytokines act.
- **Primary evidence** (doc `4928282`, section `unknown`): Here, we used theoretical considerations and experimental models of immune responses in vitro and in vivo to quantify the spatial extent of cytokine communications in dense tissues. Such heterogeneous distributions of cytokines were a source of non‐genetic cell‐to‐cell variability that is often overlooked in single‐cell studies.
- **Counter-evidence** (doc `4928282`, section `unknown`, cross-doc `False`, in-bundle `False`): The size of these self‐assembled niches scaled with the density of cytokine‐consuming cells, a parameter that gets tuned during immune responses.
- **Adjudication:** TRUE_CONTRADICTION -- `symmetric_contradiction_grounded_in_unknown`
- **Gates:** `{'counter_evidence_present': True, 'symmetric_contradiction': True, 'pool_coherent': True, 'uncontested_support': True, 'finding_grounded': True}`
- **NLI:** counter->claim P(C)=0.9691534042358398 (conf 0.9691534042358398, margin 0.9387662336230278); claim->counter P(C)=0.9665183424949646 (CONTRADICTED); primary->counter P(C)=0.0008928981842473149 (PARTIALLY_SUPPORTED); primary->claim P(C)=0.04127448797225952 (conf 0.9576387405395508)
- **Distributions:** counter `{'contradiction': 0.9691534042358398, 'neutral': 0.030387170612812042, 'entailment': 0.00045938161201775074}`; primary `{'neutral': 0.9576387405395508, 'contradiction': 0.04127448797225952, 'entailment': 0.0010867814999073744}`
- **Pool:** size 17, 1 contradictory, counter at contradiction-rank 0 / pool index 2, max P(C) in pool 0.9691534042358398
- **Why genuinely relevant:** Same variable (density of cytokine-consuming/receptor-bearing cells) and same outcome (the spatial range over which cytokines act). The claim says density has **no effect**; the counter reports niche size **scaled with** that density.

## 6. Category legend

- **A** -- semantic scope mismatch
- **B** -- unrelated evidence
- **C** -- NLI error
- **D** -- provenance / grounding failure
- **E** -- other
