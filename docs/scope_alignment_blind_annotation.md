# Blind re-annotation worksheet — scope alignment

42 cases. Machine-readable copy: `docs/scope_alignment_blind_dataset.json`.

## Why this exists

`docs/scope_alignment_failure_analysis.md` reports a near-perfect split: the
overrides it calls correct almost all invert direction on an otherwise-aligned
proposition, and the ones it calls incorrect almost all diverge on population,
intervention or outcome. **Those labels were assigned by one annotator who knew
each case's outcome and was looking for exactly that pattern.** That is a
serious bias risk, and it is the reason for this worksheet.

Here the same 42 cases are presented with every outcome signal removed, in a
different order. If the pattern is real it should reappear in labels that could
not have been fitted to it. If it does not, it was an artefact and the
hypothesis should be dropped.

## ⚠ Read this before you start

**Do not read these files first** — each would unblind you:

- `docs/scope_alignment_failure_analysis.md` / `.json` — quotes many of these
  claims verbatim *alongside their outcomes*
- `docs/scope_alignment_reconciliation.md` — lists the outcome of every case by
  claim id
- `docs/arm_f_override_error_analysis.md` / `.json`
- `eval/results_*_prosactive*.json` — the source artifacts

If you have already read any of them in detail, you are not a blind annotator
for this study and someone else should do it. Say so rather than proceeding —
a compromised re-annotation is worse than none, because it would appear to
confirm the hypothesis.

## What is withheld, and why

You see only the claim, the primary evidence, and the counter-evidence.
Withheld: the gold label, both predictions, whether the override was correct,
the adjudication status and reason, and the gate outcomes. Also withheld:
**document ids and section labels** — these correlate with the outcome in this
dataset, so showing them would partially unblind you.

Case ids (`C01`–`C42`) are assigned by a content hash, not by any ordering
that tracks the outcome. The number of correct vs incorrect overrides is not
stated.

## How to annotate

For each case, compare the **claim** against the **counter-evidence**. The
primary evidence is given for context — it is the premise the system used
before the counter-evidence was found — but your judgements are about the
claim/counter-evidence pair.

Nine fields per case:

| field | question | allowed values |
|---|---|---|
| `population_alignment` | Do the claim and the counter-evidence concern the same population, subjects, cells, or study group? | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `intervention_alignment` | Do they concern the same intervention, exposure, agent, gene, drug, or programme? | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `outcome_alignment` | Do they concern the same outcome or phenomenon? | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `endpoint_alignment` | Do they use the same specific measured endpoint? (An outcome can match while the endpoint differs -- e.g. 'cancer risk' vs 'mammographic density'.) | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `context_alignment` | Do they share the same setting or conditions -- in vivo vs in vitro, species, disease state, dose regime? | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `causal_relation_alignment` | Do they assert the same kind of relation between the same entities (causes / is required for / is associated with / interacts with)? | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `direction` | Relative to the claim, does the counter-evidence point the OPPOSITE way on a shared axis, or the SAME way? | OPPOSITE / SAME / NOT_APPLICABLE / UNCLEAR |
| `overall_scope_match` | Taken together, are the claim and the counter-evidence about the same proposition -- i.e. could one in principle confirm or refute the other? | YES / NO / UNCLEAR |
| `genuine_contradiction` | Does the counter-evidence actually contradict the claim? | YES / NO / UNCLEAR |

Plus `annotator_notes` — free text, optional. Use it whenever you pick
`UNCLEAR`, so the reason is recoverable.

### Guidance

- **`NOT_APPLICABLE` vs `DIFFERENT`.** Use `NOT_APPLICABLE` when the claim
  simply does not assert that dimension — an observational claim about
  incidence rates has no intervention. Use `DIFFERENT` only when both sides
  assert the dimension and they disagree. Conflating the two will distort the
  result.
- **`UNCLEAR` is a real answer.** Prefer it to a forced guess; a high
  `UNCLEAR` rate is itself a finding about how decidable these judgements are.
- **`direction`** asks only about orientation on a shared axis. If the two
  sentences are not about a shared axis at all, use `NOT_APPLICABLE`.
- **`overall_scope_match` and `genuine_contradiction` are independent.** Two
  sentences can be about the same proposition without contradicting (both
  agree), and a sentence can look contradictory while being about something
  else. Answer each on its own.
- **Annotate each case independently.** A few cases share their
  counter-evidence text with another case; judge each against its own claim.
- Do not try to work out what the system decided, or whether it was right.
  That information is deliberately absent and guessing at it defeats the study.

## Recording answers

Either fill the `annotation` object for each case in
`docs/scope_alignment_blind_dataset.json`, or write the table below and fill
the blanks. Leave any field you did not judge as `null` / blank rather than
guessing.

```
case_id, population, intervention, outcome, endpoint, context, causal, direction, scope_match, contradiction, notes
C01, , , , , , , , , , 
C02, , , , , , , , , , 
C03, , , , , , , , , , 
C04, , , , , , , , , , 
C05, , , , , , , , , , 
C06, , , , , , , , , , 
C07, , , , , , , , , , 
C08, , , , , , , , , , 
C09, , , , , , , , , , 
C10, , , , , , , , , , 
C11, , , , , , , , , , 
C12, , , , , , , , , , 
C13, , , , , , , , , , 
C14, , , , , , , , , , 
C15, , , , , , , , , , 
C16, , , , , , , , , , 
C17, , , , , , , , , , 
C18, , , , , , , , , , 
C19, , , , , , , , , , 
C20, , , , , , , , , , 
C21, , , , , , , , , , 
C22, , , , , , , , , , 
C23, , , , , , , , , , 
C24, , , , , , , , , , 
C25, , , , , , , , , , 
C26, , , , , , , , , , 
C27, , , , , , , , , , 
C28, , , , , , , , , , 
C29, , , , , , , , , , 
C30, , , , , , , , , , 
C31, , , , , , , , , , 
C32, , , , , , , , , , 
C33, , , , , , , , , , 
C34, , , , , , , , , , 
C35, , , , , , , , , , 
C36, , , , , , , , , , 
C37, , , , , , , , , , 
C38, , , , , , , , , , 
C39, , , , , , , , , , 
C40, , , , , , , , , , 
C41, , , , , , , , , , 
C42, , , , , , , , , , 
```

## Cases

### C01

**Claim.** Most termination events in Okazaki fragments are dictated by initiation patterns.

**Primary evidence.** Via the deep sequencing of Okazaki fragments, we comprehensively document replication fork directionality throughout the S. Using a strain in which late origins can be induced to fire early, we show that replication termination is a largely passive phenomenon that does not rely on cis-acting sequences or replication fork pausing.

**Counter-evidence.** The replication profile is predominantly determined by the kinetics of origin firing, allowing us to reconstruct chromosome-wide timing profiles from an asynchronous culture.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C02

**Claim.** Reducing H3k4me3 methylation induces mouse epiblast stem cells to naive pluripotency efficiently.

**Primary evidence.** Here, we show that blocking histone H3K4 methyltransferase MLL1 activity with the small-molecule inhibitor MM-401 reprograms mouse epiblast stem cells (EpiSCs) to naive pluripotency. These findings show that discrete perturbation of H3K4 methylation is sufficient to drive reprogramming to naive pluripotency.

**Counter-evidence.** Upon withdrawal of 2i/LIF, naive mouse ES cells can drift towards a primed pluripotent state resembling that of the post-implantation epiblast.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C03

**Claim.** Women with a lower birth weight are more likely to develop breast cancer later in life.

**Primary evidence.** We reviewed the available evidence on the association between birthweight and the risk of breast cancer. Birth weight is a significant predictor of breast cancer risk in adult life and mammary gland mass could be an intermediate stage in this long process.

**Counter-evidence.** Compared to women weighing 3001-3500 grams at birth, women with birth weights >4000g were at almost 3-fold risk of developing high mammographic density (odds ratio: 2.9, 95% confidence interval 1.1 to 7.9).

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C04

**Claim.** Citrullinated proteins externalized in neutrophil extracellular traps act indirectly to disrupt the inflammatory cycle.

**Primary evidence.** Neutrophils isolated from patients with various autoimmune diseases display enhanced neutrophil extracellular trap (NET) formation, a phenomenon that exposes autoantigens in the context of immunostimulatory molecules. CONCLUSIONS Citrullinated H4 from activated neutrophils and NETs is a target of antibodies in RA, and synthetic citrullinated H4-derived peptides are a new substrate for ACPA detection.

**Counter-evidence.** In turn, NETs significantly augmented inflammatory responses in RA and OA synovial fibroblasts, including induction of IL-6, IL-8, chemokines, and adhesion molecules.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C05

**Claim.** Charcoal is an effective treatment for acute paraquat poisoning.

**Primary evidence.** The present study retrospectively analyzed 19 patients diagnosed with paraquat (PQ) poisoning with the aim to investigate the effect of activated charcoal hemoperfusion on renal function and PQ elimination. There was a linear correlation between plasma and urine paraquat concentration prior to and during activated charcoal hemoperfusion.

**Counter-evidence.** Activated charcoal hemoperfusion had little effect on avoiding acute kidney injury in patients with severe PQ poisoning.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C06

**Claim.** Klf4 is not required for proper myeloid cell differentiation.

**Primary evidence.** In addition, inducible expression of KLF4 in the HL60 human acute myeloid leukemia cell line stimulated monocytic differentiation and enhanced 12-O-tetradecanoylphorbol 13-acetate induced macrophage differentiation, but blocked all-trans-retinoic acid induced granulocytic differentiation of HL60 cells. Simultaneous depletion of Klf2, Klf4 and Klf5 lead to ES cell differentiation.

**Counter-evidence.** The inflammation-selective effects of loss-of-KLF4 and the gain-of-KLF4-induced monocytic differentiation in HL60 cells identify KLF4 as a key regulator of monocytic differentiation and a potential target for translational immune modulation.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C07

**Claim.** Diminished ovarian reserve is a reliable indicator of infertility in a non-infertile population.

**Primary evidence.** Objective To determine the associations between biomarkers of ovarian reserve and reproductive potential among women of late reproductive age. Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility.

**Counter-evidence.** Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C08

**Claim.** Cholesterol loading induces KLF4 expression in VSMCs, resulting in the expression of pro-inflammatory cytokines.

**Primary evidence.** We used a pan-DC, CD11c-specific cre-lox gene knockout mouse model to assess the role of KLF2 in DC activation, function, and control of inflammation in the context of hypercholesterolemia and atherosclerosis. Membrane cholesterol modulates a variety of cell signaling pathways and functions.

**Counter-evidence.** Here we show overt pro-inflammatory effects of HDL-mediated passive cholesterol depletion and lipid raft disruption in murine and human primary macrophages in vitro.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C09

**Claim.** Patients with common epithelial cancers are less likely to have an emergency event as their first hospital admission if they live in resource-deprived areas.

**Primary evidence.** RESULTS Patients from deprived areas, older people, and women were more likely to be admitted as emergencies. Patients with colonic cancer were very significantly less likely to receive any treatment if they came from any but the most affluent area (ORs 0.639, 0.603 and 0.544 in increasingly deprived quartiles), this may have been exacerbated if the hospital was distant from their residence (OR for forth quartile for both travel and deprivation 0.731, not significant).

**Counter-evidence.** RESULTS Patients from deprived areas, older people, and women were more likely to be admitted as emergencies.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C10

**Claim.** Human embryonic stem cells give rise to cell types from the outer embryonic germ layer, but not the other two layers.

**Primary evidence.** When grafted into SCID mice, both lines give rise to teratomas containing derivatives of all three embryonic germ layers. Embryonic stem cells provide a model to study early human embryology, an investigational tool for discovery of novel growth factors and medicines, and a potential source of cells for use in transplantation therapy.

**Counter-evidence.** After undifferentiated proliferation in vitro for 4 to 5 months, these cells still maintained the developmental potential to form trophoblast and derivatives of all three embryonic germ layers, including gut epithelium (endoderm); cartilage, bone, smooth muscle, and striated muscle (mesoderm); and neural epithelium, embryonic ganglia, and stratified squamous epithelium (ectoderm).

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C11

**Claim.** Bariatric surgery increases rates of postmenopausal breast cancer.

**Primary evidence.** Bariatric surgery was, however, consistently associated with postoperative decreases in the prevalence of depression (7 studies; 8%-74% decrease) and the severity of depressive symptoms (6 studies; 40%-70% decrease). Widening the availability of bariatric surgery could lead to substantial health b

**Counter-evidence.** No association was detected between bariatric surgery and fractures, cancer, or stroke.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C12

**Claim.** CD28 initiates tonic signaling in conventional T cells, which causes an exhaustion phenotype and limited efficiency.

**Primary evidence.** CD28 is one of the molecules that provide costimulatory signals for T cells. To our knowledge, this is the first report describing an essential role for CD28 in protective immunity against an intracellular microbial pathogen.

**Counter-evidence.** We further determine that CD28 costimulation augments, whereas 4-1BB costimulation reduces, exhaustion induced by persistent CAR signaling.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C13

**Claim.** Commelina yellow mottle virus (ComYMV) has three typical badnavirus ORFs, capable of encoding proteins 23, 15, and 216 kD.

**Primary evidence.** We have characterized the viral transcript and determined the complete sequence of the genome of Commelina mellow mottle virus (CoYMV), a member of this group. A new badnavirus was discovered from pagoda trees showing yellow mosaic symptoms on the leaves by high throughput sequencing of small RNAs.

**Counter-evidence.** Analysis of the viral transcript indicates that the virus encodes a single terminally-redundant genome-length plus 120 nucleotide transcript.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C14

**Claim.** Deltex has no known interactions with eIF3f.

**Primary evidence.** We show that eIF3f has an intrinsic DUB activity. Knocking down eIF3f leads to an accumulation of monoubiquitinated forms of activated Notch, an effect counteracted by murine WT eIF3f but not by a catalytically inactive mutant.

**Counter-evidence.** We also show that eIF3f is recruited to activated Notch on endocytic vesicles by the putative E3 ubiquitin ligase Deltex1, which serves as a bridging factor.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C15

**Claim.** Pyridostatin reduces proliferation of homologous recombination - defective cells.

**Primary evidence.** Treatment with the G4-stabilizing compound pyridostatin (PDS) increases telomere fragility in BRCA2-deficient cells, suggesting that G4 formation drives telomere instability. BLM, WRN, and p53 are involved in the homologous DNA recombination pathway.

**Counter-evidence.** Treatment with the G4-stabilizing compound pyridostatin (PDS) increases telomere fragility in BRCA2-deficient cells, suggesting that G4 formation drives telomere instability.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C16

**Claim.** Cognitive behavioral therapy is an ineffective treatment for insomnia

**Primary evidence.** OBJECTIVE To examine short- and long-term clinical efficacy of cognitive behavioral therapy (CBT) and pharmacological treatment in older adults experiencing chronic primary insomnia. Data from this study on the treatment of chronic, severe insomnia (a small subset of all insomnia) are not generalizable to the typical insomnia patient.

**Counter-evidence.** RESULTS CBT resulted in improved short- and long-term outcomes compared with zopiclone on 3 out of 4 outcome measures.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C17

**Claim.** Removal of H3K9me3 by ectopic expression of other H3K9 demethylases decreases reprogramming efficiency in SCNT experiments.

**Primary evidence.** Here we show that overexpression of a related H3K9me3 demethylase KDM4A improves human SCNT, and that, as in mice, H3K9me3 in the human somatic cell genome is an SCNT reprogramming barrier. Furthermore, we show that interactions between RNA and KDM4E are essential for H3K9 demethylation during EGA.

**Counter-evidence.** Here we show that overexpression of a related H3K9me3 demethylase KDM4A improves human SCNT, and that, as in mice, H3K9me3 in the human somatic cell genome is an SCNT reprogramming barrier.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C18

**Claim.** Bariatric surgery increases rates of colorectal cancer.

**Primary evidence.** Widening the availability of bariatric surgery could lead to substantial health b Bariatric surgery was, however, consistently associated with postoperative decreases in the prevalence of depression (7 studies; 8%-74% decrease) and the severity of depressive symptoms (6 studies; 40%-70% decrease).

**Counter-evidence.** No association was detected between bariatric surgery and fractures, cancer, or stroke.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C19

**Claim.** Medications to treat obesity are highly effective.

**Primary evidence.** To succeed in developing drugs that control body weight to the extent seen following surgical intervention, it seems obvious that a new paradigm is needed. OBJECTIVE To summarise the long term efficacy of anti-obesity drugs in reducing weight and improving health status.

**Counter-evidence.** It is therefore unsurprising that anti-obesity drug discovery programmes have been littered with false starts, failures in clinical development, and withdrawals due to adverse effects that were not fully appreciated at the time of launch.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C20

**Claim.** Human embryonic stem cells give rise to cell types from all three embryonic germ layers.

**Primary evidence.** When grafted into SCID mice, both lines give rise to teratomas containing derivatives of all three embryonic germ layers. Embryonic stem cells provide a model to study early human embryology, an investigational tool for discovery of novel growth factors and medicines, and a potential source of cells for use in transplantation therapy.

**Counter-evidence.** Both cell lines differentiate in vitro into extraembryonic and somatic cell lineages.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C21

**Claim.** G-CSF increases the expansion and infiltration of MDSCs into tumors.

**Primary evidence.** Inhibiting this pathway or its activators (for example, FGFR) impairs tumour progression, which is partially rescued by restoring MDSCs or G-CSF. Tumour-initiating cells (TICs) exhibit elevated G-CSF.

**Counter-evidence.** CSF1 produced by tumor cells caused HDAC2-mediated downregulation of granulocyte-specific chemokine expression in CAF, which limited migration of these cells to tumors.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C22

**Claim.** Notch signaling occurs between tumor cells and stromal cells.

**Primary evidence.** We recently demonstrated that Notch signaling is involved in proliferation and survival of B cell-derived tumor cells of classic Hodgkin disease and described a novel mechanism for the oncogenic capacity of Notch. In this study we investigated whether Notch signaling is involved in the tight interactions between neoplastic plasma cells and their bone marrow microenvironment, which are essential for tumor cell growth in multiple myeloma (MM).

**Counter-evidence.** Notch receptors expressed on hematopoietic stem cells interact with their ligands on bone marrow stromal cells and thereby control cell fate decisions and survival.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C23

**Claim.** Taking 400mg of α-tocopheryl acetate in combination with vitamin C reduces the risk of prostate cancer.

**Primary evidence.** CONCLUSIONS In this large, long-term trial of male physicians, neither vitamin E nor C supplementation reduced the risk of prostate or total cancer. Some human and animal studies indicate that plasma concentrations of gamma-tocopherol are inversely associated with the incidence of cardiovascular disease and prostate cancer.

**Counter-evidence.** CONCLUSIONS In this large, long-term trial of male physicians, neither vitamin E nor C supplementation reduced the risk of prostate or total cancer.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C24

**Claim.** Pioneer factor OCT3/4 interacts with major chromatin remodeling factors.

**Primary evidence.** Purified Oct4, Sox2, and Klf4 proteins can bind nucleosomes in vitro, and in vivo they preferentially target silent sites enriched for nucleosomes. Our findings provide insight into how pioneer factors can target naive chromatin sites.

**Counter-evidence.** FoxA, the paradigm pioneer TF, has a winged helix DBD that resembles linker histone and thereby binds its target sites on nucleosomes and in compacted chromatin.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C25

**Claim.** Systemic immunosuppressive (IS) therapy increases the chance of cancer mortality in patients with inflammatory eye disease (IED)

**Primary evidence.** DESIGN Retrospective cohort study evaluating overall and cancer mortality in relation to immunosuppressive drug exposure among patients with ocular inflammatory diseases. For patients unexposed to immunosuppressive treatment, risks of death overall (standardised mortality ratio 1.02, 95% confidence interval [CI] 0.94 to 1.11) and from cancer (1.10, 0.93 to 1.29) were similar to those of the US population.

**Counter-evidence.** Patients who used azathioprine, methotrexate, mycophenolate mofetil, ciclosporin, systemic corticosteroids, or dapsone had overall and cancer mortality similar to that of patients who never took immunosuppressive drugs.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C26

**Claim.** The DESMOND program demonstrates no significant impact on biochemical outcomes.

**Primary evidence.** The overall evidence rating was low to very low for important outcomes such as pre-eclampsia, gestational diabetes, gestational hypertension, and preterm delivery. No patients received ALK inhibitor therapy.

**Counter-evidence.** Dietary intervention resulted in the largest reduction in maternal gestational weight gain (3.84 kg, 2.45 to 5.22 kg), with improved pregnancy outcomes compared with other interventions.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C27

**Claim.** 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.

**Primary evidence.** Patients receiving methadone maintenance are accepted by 56% of the responding programs. Approximately 180 patients receiving methadone maintenance are reported to have undergone liver transplantation.

**Counter-evidence.** Ninety-four percent have addiction treatment requirements.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C28

**Claim.** PKG-la does not have a large impact on expression of spinal long term potentiation in PGK-la knockout mice.

**Primary evidence.** Here, we found that Protein Kinase G1 (PKG-I) localized presynaptically in nociceptor terminals plays an essential role in the expression of spinal LTP. Furthermore, we show that the knockout mice exhibited significant abnormalities in spine morphology and in synaptic function, including enhanced hippocampal long-term potentiation.

**Counter-evidence.** Patch clamp recordings showed that activity-induced LTP at identified synapses between nociceptors and spinal neurons projecting to the periaqueductal grey (PAG) was completely abolished in SNS-PKG-I(-/-) mice, although basal synaptic transmission was not affected.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C29

**Claim.** Incidence of heart failure increased by 10% in women since 1979.

**Primary evidence.** RESULTS The incidence of heart failure was higher among men (378/100 000 persons; 95% confidence interval [CI], 361-395 for men; 289/100 000 persons; 95% CI, 277-300 for women) and did not change over time among men or women. CONCLUSION In this community-based cohort, the incidence of heart failure has not declined during 2 decades, but survival after onset of heart failure has increased overall, with less improvement among women and elderly persons.

**Counter-evidence.** RESULTS The incidence of heart failure was higher among men (378/100 000 persons; 95% confidence interval [CI], 361-395 for men; 289/100 000 persons; 95% CI, 277-300 for women) and did not change over time among men or women.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C30

**Claim.** Repressing IL-18 has negative effects on atherosclerotic lesion composition and progression.

**Primary evidence.** IL-18 was highly expressed in the atherosclerotic plaques compared with control normal arteries and was localized mainly in plaque macrophages. CONCLUSIONS These results suggest, for the first time, a major role for IL-18 in atherosclerotic plaque destabilization leading to acute ischemic syndromes.

**Counter-evidence.** Interruption of IL18 action reduces atherosclerosis in mice.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C31

**Claim.** Pre-mRNAs associated with spliceosomal components are less stable than unassociated splicing substrates.

**Primary evidence.** In order to provide both accuracy to the recognition of reactive splice sites in the pre-mRNA and flexibility to the choice of splice sites during alternative splicing, the spliceosome exhibits exceptional compositional and structural dynamics that are exploited during substrate-dependent complex assembly, catalytic activation, and active site remodeling. Furthermore, the process of transcription influences alternative splicing of newly synthesized pre-mRNAs.

**Counter-evidence.** Pol II-synthesized RNAs containing functional splice sites are protected from nuclear degradation, presumably because the local concentration of the splicing machinery is sufficiently high to ensure its association over interactions with nucleases.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C32

**Claim.** Cost effectiveness evaluations based on cRCT data accurately reflect prices for patients in actual clinical practice.

**Primary evidence.** The objective of this study was to evaluate the external validity of published cost-effectiveness studies by comparing the data used in these studies (typically based on RCTs) to observational data from actual clinical practice. CONCLUSION Evaluating the cost effectiveness of prognostic biomarkers is important even when effects at an individual level are small.

**Counter-evidence.** CONCLUSIONS The published cost-effectiveness analyses of coxibs lacked external validity, did not represent patients in actual clinical practice, and should not have been used to inform prescribing policies.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C33

**Claim.** The density of cytokine receptor bearing cells has no effect on the distance over which cytokines act.

**Primary evidence.** Here, we used theoretical considerations and experimental models of immune responses in vitro and in vivo to quantify the spatial extent of cytokine communications in dense tissues. Such heterogeneous distributions of cytokines were a source of non‐genetic cell‐to‐cell variability that is often overlooked in single‐cell studies.

**Counter-evidence.** The size of these self‐assembled niches scaled with the density of cytokine‐consuming cells, a parameter that gets tuned during immune responses.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C34

**Claim.** The recruitment of Wdr5 to its target loci depends on Kat8.

**Primary evidence.** FISH showed that, although they were heterogeneous, breakpoints on 9p resulted in the partial or complete deletion of PAX5. This approach identified 5 fusion gene partners with PAX5: LOC392027 (7p12.1), SLCO1B3 (12p12), ASXL1 (20q11.1), KIF3B (20q11.21), and C20orf112 (20q11.1).

**Counter-evidence.** FISH showed that, although they were heterogeneous, breakpoints on 9p resulted in the partial or complete deletion of PAX5.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C35

**Claim.** Splenomegaly is observed in knockin mouse lacking the SHP-2 MAPK pathway.

**Primary evidence.** The SHP2 signal-deficient mice (gp130F759/F759 were born normal but displayed splenomegaly and lymphadenopathy and an enhanced acute phase reaction. Neural cell-fate decisions were similarly perturbed in a mouse knockin model that phenocopies human NS.

**Counter-evidence.** Here, we demonstrate that genetic knockdown of SHP-2 in cultured cortical precursors or in the embryonic cortex inhibited basal neurogenesis and caused enhanced and precocious astrocyte formation.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C36

**Claim.** The DESMOND program demonstrates no significant impact on lifestyles outcomes.

**Primary evidence.** With all interventions combined, there were no significant differences in birth weight (mean difference -50 g, -100 to 0 g) and the incidence of large for gestational age (relative risk 0.85, 0.66 to 1.09) or small for gestational age (1.00, 0.78 to 1.28) babies between the groups, though by itself physical activity was associated with reduced birth weight (mean difference -60 g, -120 to -10 g). The overall evidence rating was low to very low for important outcomes such as pre-eclampsia, gestational diabetes, gestational hypertension, and preterm delivery.

**Counter-evidence.** Dietary intervention resulted in the largest reduction in maternal gestational weight gain (3.84 kg, 2.45 to 5.22 kg), with improved pregnancy outcomes compared with other interventions.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C37

**Claim.** The risk of cancer rises with level of alcohol consumption.

**Primary evidence.** RESULTS Drinkers of wine outside meals exhibited higher death rates from all causes, noncardiovascular diseases, and cancer, as compared to drinkers of wine with meals. It has been hypothesized, and preliminary data support the notion, that drinking pattern could have important influences on determining the health effects of alcohol (13,14).

**Counter-evidence.** Wine, particularly red wine, has high levels of phenolic compounds that favorably influence multiple biochemical systems, such as increased high-density lipoprotein cholesterol, antioxidant activity, decreased platelet aggregation and endothelial adhesion, suppression of cancer cell growth, and promotion of nitric oxide production.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C38

**Claim.** Asymptomatic visual impairment screening in elderly populations leads to improved vision.

**Primary evidence.** OBJECTIVE To assess whether population screening for impaired vision among older people in the community leads to improvements in vision. Visual impairment in this age group can usually be reduced with treatment.

**Counter-evidence.** CONCLUSIONS Screening of asymptomatic older people in the community is not justified on present evidence.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C39

**Claim.** Chenodeoxycholic acid treatment decreases brown adipose tissue activity.

**Primary evidence.** Brown fat is also able to utilize blood glucose and lipid and results in improved glucose metabolism and blood lipid independent of weight loss. The inducible brown adipocyte, beige adipocyte evolving by thermogenic activation of white adipose tissue have different origin and molecular signature from classical brown adipocytes but share the characteristics of high mitochondria content, UCP1 expression and thermogenic capacity when activated.

**Counter-evidence.** Treatment of 12 healthy female subjects with CDCA for 2 days resulted in increased BAT activity.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C40

**Claim.** The peak incidence of fractures occurs in toddlers.

**Primary evidence.** CONCLUSIONS There has been a statistically significant increase in the incidence of distal forearm fractures in children and adolescents, but whether this is due to changing patterns of physical activity, decreased bone acquisition due to poor calcium intake, or both is unclear at present. Given the large number of childhood fractures, however, studies are needed to define the cause(s) of this increase.

**Counter-evidence.** CONTEXT The incidence of distal forearm fractures in children peaks around the time of the pubertal growth spurt, possibly because physical activity increases at the time of a transient deficit in cortical bone mass due to the increased calcium demand during maximal skeletal growth.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C41

**Claim.** A diminished ovarian reserve is a very strong indicator of infertility, even in an a priori non-infertile population.

**Primary evidence.** Objective To determine the associations between biomarkers of ovarian reserve and reproductive potential among women of late reproductive age. Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility.

**Counter-evidence.** Conclusions and Relevance Among women aged 30 to 44 years without a history of infertility who had been trying to conceive for 3 months or less, biomarkers indicating diminished ovarian reserve compared with normal ovarian reserve were not associated with reduced fertility.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

### C42

**Claim.** Natriuretic peptides increase susceptibility to diabetes.

**Primary evidence.** Further studies are needed to investigate the mechanisms underlying this association and possibilities for preventive interventions. This review considers research in the field of type 1 diabetes toward identifying disease mechanisms using genetic approaches.

**Counter-evidence.** In participants of our case-cohort study who were free of T2D and cardiovascular disease at baseline, we observed a 21% (95% CI 3%-36%) decreased risk of incident T2D per one standard deviation (SD) higher log-transformed NT-pro-BNP levels in analysis adjusted for age, sex, body mass index, systolic blood pressure, smoking, family history of T2D, history of hypertension, and levels of triglycerides, high-density lipoprotein cholesterol, and low-density lipoprotein cholesterol.

| field | answer |
|---|---|
| `population_alignment` | |
| `intervention_alignment` | |
| `outcome_alignment` | |
| `endpoint_alignment` | |
| `context_alignment` | |
| `causal_relation_alignment` | |
| `direction` | |
| `overall_scope_match` | |
| `genuine_contradiction` | |
| `annotator_notes` | |

---

## After annotation

Hand the completed file back for scoring. The comparison against the recorded
outcomes has not been designed yet and is deliberately out of scope here, so
no metric is defined in this document — that keeps the annotation free of any
hint about what pattern would count as confirmation.
