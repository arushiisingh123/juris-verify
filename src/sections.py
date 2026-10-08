"""
Scientific section / context detection for Juris-Verify evidence ranking.

WHY THIS EXISTS
---------------
The N=504 SciFact error analysis attributes ~84% of missed contradictions to
evidence *selection*: the source document contains a sentence that the existing
NLIVerifier scores as CONTRADICTED, but `extract_best_evidence` instead returns
a high-lexical-overlap Background/Objective sentence. That is a predictable
failure mode -- an "OBJECTIVE To determine whether X reduces Y" sentence
restates the claim almost verbatim (so cosine similarity loves it) while
carrying no finding at all, whereas the sentence that actually conflicts with
the claim lives under RESULTS or CONCLUSIONS and is phrased in outcome
language.

This module recovers *where in the paper* a candidate sentence came from, so
the experimental ranker in src/extraction.py can express the prior "a reported
finding outranks a restatement of the research question".

DOES SCIFACT EXPOSE SECTION HEADINGS?
-------------------------------------
Not as a separate metadata field. `src/data_loader.build_document_corpus`
flattens each corpus record to `title + " " + " ".join(abstract_sentences)`,
and the HF `allenai/scifact` corpus schema carries no per-sentence section
label.

But structured abstracts in that corpus carry their headings *inline*, as an
ALL-CAPS prefix on the sentence that opens each section. Verbatim examples from
the tracked artifact eval/results_20260907_095418_n504.json:

    "OBJECTIVE To determine whether high doses of folic acid and B vitamins ..."
    "RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP ..."
    "DESIGN, SETTING, AND PATIENTS Retrospective analysis of German and US ..."
    "CONCLUSIONS A dosage of folic acid of 0.8 mg/d appears necessary ..."

So we use the headings the documents really do expose, and fall back to
"unknown" wherever they do not. The extraction below is a deterministic,
documented heuristic: no model, no learned component, no per-claim special
cases.

THE HEURISTIC, EXACTLY
----------------------
For one document's sentence list, in source order:

1. Take the sentence's leading run of "heading-shaped" tokens. A token is
   heading-shaped iff, after stripping trailing ":" and "," characters, it is
   non-empty and every character is upper-case alphanumeric or one of & / - or
   an apostrophe. Upper case is required, so "Results showed ..." is NOT a
   heading while "RESULTS Of the ..." is. The run stops at the first token
   that is not heading-shaped, or after MAX_HEADING_TOKENS tokens.
2. Normalize that run: strip ":" / "," from each token, join with single spaces.
3. Longest-match the normalized run against SECTION_HEADINGS, trying the full
   n-token phrase first and shortening from the right. The first (longest)
   match wins; its value is the sentence's section.
      - "METHODS AND RESULTS We ..."  -> 3-gram hits -> "results"
      - "CONCLUSIONS A dosage ..."    -> 2-gram misses, 1-gram hits -> "conclusions"
4. No match -> this sentence does not open a section, and it INHERITS the
   section of the most recent heading seen earlier in the same document
   ("sticky" carry-forward). Only the first sentence of a section carries the
   heading, so carry-forward is what labels the rest of that section.
5. Sentences before any heading in that document -- and every sentence of an
   unstructured abstract -- are SECTION_UNKNOWN.

Carry-forward is per-document and never crosses a document boundary.

KNOWN LIMITATION: `split_sentences` in src/extraction.py drops fragments of 20
characters or fewer. A structured abstract that put a heading on its own short
line (e.g. "RESULTS") rather than inline would therefore lose that heading and
fall back to carry-forward / unknown. SciFact headings are inline, so this does
not arise in this corpus, but it is the failure shape to expect on a corpus
that formats headings differently.

WHY A CLOSED HEADING VOCABULARY
-------------------------------
A permissive "any ALL-CAPS prefix is a heading" rule mislabels biomedical
acronyms that legitimately open sentences. An audit of the N=504 artifact found
PTEN, IFI16, IL-10, IL-18, AMPK, CD28, EBI2, FGF21, DUSP4, PDPN, HAND2, DNA and
FISH all matching such a rule. Matching against the fixed vocabulary below
rejects every one of them, at the cost of silently returning "unknown" for an
unlisted heading -- which is the safe direction, because "unknown" carries a
neutral (zero) prior and therefore cannot change the ranking.
"""

# --- section classes --------------------------------------------------------
# Deliberately coarse: four classes plus "unknown". The ranking prior only
# needs to answer "is this a reported finding or a restatement of the
# question", and coarse classes keep the resulting score readable.
SECTION_BACKGROUND = "background"
SECTION_METHODS = "methods"
SECTION_RESULTS = "results"
SECTION_CONCLUSIONS = "conclusions"
SECTION_UNKNOWN = "unknown"

SECTION_CLASSES = (
    SECTION_BACKGROUND,
    SECTION_METHODS,
    SECTION_RESULTS,
    SECTION_CONCLUSIONS,
    SECTION_UNKNOWN,
)

# --- ordinal ranking prior, in [-1.0, +1.0] ---------------------------------
# This table carries the *direction and relative strength* of the preference; a
# single scalar (`section_weight` in src/extraction.py) carries its magnitude.
# Keeping the two separate means one knob controls how much section information
# matters overall, and setting that knob to 0.0 disables this module's
# influence exactly.
#
#   +1.0 results / conclusions -- sentences that report an outcome. A
#        contradiction of a factual claim is almost always stated here.
#    0.0 unknown -- no heading recovered; stay neutral, so an unstructured
#        abstract is never penalized just for being unstructured.
#   -0.5 methods -- describes how the study was run; high topical overlap with
#        the claim, low chance of resolving it. Penalized, but less than
#        background, because a Methods sentence at least reports concrete
#        study facts rather than the question itself.
#   -1.0 background -- Background / Objective / Introduction. The documented
#        worst case: restates the research question in the claim's own words.
SECTION_PRIORS = {
    SECTION_RESULTS: 1.0,
    SECTION_CONCLUSIONS: 1.0,
    SECTION_UNKNOWN: 0.0,
    SECTION_METHODS: -0.5,
    SECTION_BACKGROUND: -1.0,
}

# The longest headings observed in the corpus audit are 4 tokens
# ("DESIGN, SETTING, AND PATIENTS"); 6 leaves room for unseen variants without
# letting an all-upper-case sentence run away.
MAX_HEADING_TOKENS = 6

_HEADING_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&/-'")

# --- heading vocabulary -----------------------------------------------------
# Normalized (upper-case, punctuation-stripped, single-spaced) heading phrase
# -> section class. Compound headings are listed explicitly and resolve to the
# *outcome-bearing* half, because that is what the sentence under them reports
# (a "METHODS AND RESULTS" sentence states results).
SECTION_HEADINGS = {
    # background / objective / introduction
    "BACKGROUND": SECTION_BACKGROUND,
    "BACKGROUNDS": SECTION_BACKGROUND,
    "CONTEXT": SECTION_BACKGROUND,
    "INTRODUCTION": SECTION_BACKGROUND,
    "IMPORTANCE": SECTION_BACKGROUND,
    "OBJECTIVE": SECTION_BACKGROUND,
    "OBJECTIVES": SECTION_BACKGROUND,
    "AIM": SECTION_BACKGROUND,
    "AIMS": SECTION_BACKGROUND,
    "PURPOSE": SECTION_BACKGROUND,
    "GOAL": SECTION_BACKGROUND,
    "GOALS": SECTION_BACKGROUND,
    "RATIONALE": SECTION_BACKGROUND,
    "HYPOTHESIS": SECTION_BACKGROUND,
    "BACKGROUND AND AIM": SECTION_BACKGROUND,
    "BACKGROUND AND AIMS": SECTION_BACKGROUND,
    "BACKGROUND AND OBJECTIVE": SECTION_BACKGROUND,
    "BACKGROUND AND OBJECTIVES": SECTION_BACKGROUND,
    "BACKGROUND AND PURPOSE": SECTION_BACKGROUND,
    "OBJECTIVE AND BACKGROUND": SECTION_BACKGROUND,
    # methods / design
    "METHOD": SECTION_METHODS,
    "METHODS": SECTION_METHODS,
    "METHODOLOGY": SECTION_METHODS,
    "MATERIALS AND METHODS": SECTION_METHODS,
    "PATIENTS AND METHODS": SECTION_METHODS,
    "SUBJECTS AND METHODS": SECTION_METHODS,
    "DESIGN": SECTION_METHODS,
    "STUDY DESIGN": SECTION_METHODS,
    "DESIGN AND SETTING": SECTION_METHODS,
    "DESIGN SETTING AND PARTICIPANTS": SECTION_METHODS,
    "DESIGN SETTING AND PATIENTS": SECTION_METHODS,
    "SETTING": SECTION_METHODS,
    "SETTINGS": SECTION_METHODS,
    "SETTING AND PARTICIPANTS": SECTION_METHODS,
    "PARTICIPANT": SECTION_METHODS,
    "PARTICIPANTS": SECTION_METHODS,
    "PATIENT": SECTION_METHODS,
    "PATIENTS": SECTION_METHODS,
    "SUBJECTS": SECTION_METHODS,
    "INTERVENTION": SECTION_METHODS,
    "INTERVENTIONS": SECTION_METHODS,
    "EXPOSURE": SECTION_METHODS,
    "EXPOSURES": SECTION_METHODS,
    "MEASUREMENT": SECTION_METHODS,
    "MEASUREMENTS": SECTION_METHODS,
    "MEASURES": SECTION_METHODS,
    "MAIN OUTCOME MEASURE": SECTION_METHODS,
    "MAIN OUTCOME MEASURES": SECTION_METHODS,
    "MAIN OUTCOMES AND MEASURES": SECTION_METHODS,
    "OUTCOME MEASURES": SECTION_METHODS,
    "DATA SOURCES": SECTION_METHODS,
    "DATA COLLECTION": SECTION_METHODS,
    "DATA EXTRACTION": SECTION_METHODS,
    "DATA SYNTHESIS": SECTION_METHODS,
    "STUDY SELECTION": SECTION_METHODS,
    "ELIGIBILITY CRITERIA": SECTION_METHODS,
    "STATISTICAL ANALYSIS": SECTION_METHODS,
    "ANALYSIS": SECTION_METHODS,
    # results / findings
    "RESULT": SECTION_RESULTS,
    "RESULTS": SECTION_RESULTS,
    "FINDING": SECTION_RESULTS,
    "FINDINGS": SECTION_RESULTS,
    "OUTCOME": SECTION_RESULTS,
    "OUTCOMES": SECTION_RESULTS,
    "OBSERVATIONS": SECTION_RESULTS,
    "METHODS AND RESULTS": SECTION_RESULTS,
    "MATERIALS AND RESULTS": SECTION_RESULTS,
    "MEASUREMENTS AND RESULTS": SECTION_RESULTS,
    "RESULTS AND DISCUSSION": SECTION_RESULTS,
    "RESULTS AND CONCLUSION": SECTION_RESULTS,
    "RESULTS AND CONCLUSIONS": SECTION_RESULTS,
    # conclusions / interpretation
    "CONCLUSION": SECTION_CONCLUSIONS,
    "CONCLUSIONS": SECTION_CONCLUSIONS,
    "CONCLUSION AND RELEVANCE": SECTION_CONCLUSIONS,
    "CONCLUSIONS AND RELEVANCE": SECTION_CONCLUSIONS,
    "CONCLUSIONS AND IMPLICATIONS": SECTION_CONCLUSIONS,
    "INTERPRETATION": SECTION_CONCLUSIONS,
    "INTERPRETATIONS": SECTION_CONCLUSIONS,
    "DISCUSSION": SECTION_CONCLUSIONS,
    "IMPLICATION": SECTION_CONCLUSIONS,
    "IMPLICATIONS": SECTION_CONCLUSIONS,
    "SIGNIFICANCE": SECTION_CONCLUSIONS,
    "SUMMARY": SECTION_CONCLUSIONS,
}


def _heading_token_run(sentence):
    """Return the leading run of heading-shaped tokens, normalized.

    Returns:
        list[str]: up to MAX_HEADING_TOKENS upper-case tokens with trailing
        ":" / "," stripped; empty when the sentence does not start with a
        heading-shaped token.
    """
    run = []
    for raw in sentence.split():
        if len(run) >= MAX_HEADING_TOKENS:
            break
        token = raw.rstrip(":,")
        if not token:
            break
        if any(ch not in _HEADING_TOKEN_CHARS for ch in token):
            break
        run.append(token)
    return run


def detect_section_heading(sentence):
    """Detect the section heading that `sentence` opens, if any.

    Deterministic and side-effect free: same input, same output, always.

    Args:
        sentence: a candidate sentence (verbatim document text)

    Returns:
        tuple: (section, heading) where `section` is one of SECTION_CLASSES
        other than SECTION_UNKNOWN and `heading` is the normalized heading
        phrase that matched -- or (None, None) when this sentence does not open
        a section.
    """
    if not sentence:
        return None, None

    run = _heading_token_run(sentence)

    # Longest match first: shorten from the right until a phrase is known.
    for end in range(len(run), 0, -1):
        phrase = " ".join(run[:end])
        section = SECTION_HEADINGS.get(phrase)
        if section is not None:
            return section, phrase

    return None, None


def assign_sections(sentences):
    """Label each sentence of ONE document with its section class.

    Implements the sticky carry-forward described in the module docstring: a
    sentence that opens a section takes that section, every following sentence
    inherits it, and anything before the first heading is SECTION_UNKNOWN.

    Args:
        sentences: list of sentences from a single document, in source order

    Returns:
        list[str]: one section class per input sentence, same length and order.
    """
    sections = []
    current = SECTION_UNKNOWN

    for sentence in sentences:
        section, _heading = detect_section_heading(sentence)
        if section is not None:
            current = section
        sections.append(current)

    return sections


def section_prior(section):
    """Ordinal ranking prior for a section class, in [-1.0, +1.0].

    Unrecognized or missing section metadata maps to 0.0 (neutral), so a
    document whose structure could not be read is never penalized.
    """
    return SECTION_PRIORS.get(section, 0.0)
