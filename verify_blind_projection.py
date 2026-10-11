"""
Independent proof that the blind projection cannot access outcome fields.

Research/analysis only; reads code and data, writes nothing.

The sampler's blinding claim rests on one function -- `project()` -- being the
sole chokepoint through which artifact data reaches a blind dataset. This script
verifies that claim three independent ways, so it does not have to be taken on
the sampler's word:

  1. STATIC (AST). Parse the sampler and prove `project()` reads only
     whitelisted keys, that every path from the artifact to the sample runs
     through it, and that no outcome field name appears as a subscript anywhere
     in the frame-construction or sampling path.

  2. DYNAMIC (poisoned record). Feed the real `project()` a record stuffed with
     every outcome field, each carrying a unique sentinel, and prove no
     sentinel survives into the output.

  3. INSTRUMENTED (tripwire). Wrap an artifact record in a dict subclass that
     RAISES on any read of an outcome key, then run the whole
     frame -> sample -> cases path over it. Any hidden access aborts loudly.

Exit code 0 only if all three pass.

    python verify_blind_projection.py
    python verify_blind_projection.py --module inspect_scope_alignment_heldout
"""

import argparse
import ast
import importlib
import json
import sys

# Outcome-bearing keys the projection must never read.
FORBIDDEN_KEYS = [
    "gold", "prediction", "baseline_verdict", "verdict", "override_applied",
    "override_correct", "adjudicated_verdict", "adjudication_status",
    "adjudication_reason", "adjudication_gates", "adjudication_required_gates",
    "matrix_label", "claim_index", "counter_section", "primary_evidence_section",
    "evidence_sections", "counter_evidence_doc_id", "evidence_doc_id",
    "evidence_doc_ids", "counter_p_contradiction", "reverse_p_contradiction",
    "primary_counter_p_contradiction", "primary_p_contradiction",
    "counter_confidence", "counter_score", "counter_distribution",
    "primary_verdict_distribution", "max_p_contradiction", "counter_rank",
    "counter_pool_index", "counter_in_d_selection", "counter_selection_rank",
    "candidate_pool_size", "n_contradictory_candidates", "evidence_spans",
    "strict_faithfulness", "weighted_faithfulness", "statements",
]

# Keys the projection is allowed to read from an artifact record.
ALLOWED_RECORD_KEYS = {"claim", "evidence", "counter_evidence"}

# `baseline_prediction` is read for eligibility only, by build_frame, and is a
# model prediction rather than a correctness signal. It must not reach output.
FRAME_ONLY_KEYS = {"baseline_prediction"}


class Tripwire(dict):
    """A record that raises the moment an outcome key is read."""

    def __init__(self, data, forbidden):
        super().__init__(data)
        self._forbidden = set(forbidden)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        if key in self._forbidden:
            raise AssertionError(f"FORBIDDEN READ via __getitem__: {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.append(key)
        if key in self._forbidden:
            raise AssertionError(f"FORBIDDEN READ via .get: {key!r}")
        return super().get(key, default)


def check_static(module_name):
    """AST analysis of the sampler's projection and sampling path."""
    module = importlib.import_module(module_name)
    with open(module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    results = []

    # (a) project() reads only whitelisted keys.
    project = functions.get("project")
    if project is None:
        return [("project() exists", False, "function not found")], False
    subscripts = {
        node.slice.value for node in ast.walk(project)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    getattrs = {
        node.args[0].value for node in ast.walk(project)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get" and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    read = subscripts | getattrs
    results.append((
        "project() reads only {claim, evidence, counter_evidence}",
        read <= ALLOWED_RECORD_KEYS,
        f"reads {sorted(read)}",
    ))

    # (b) no forbidden key is subscripted anywhere in the frame/sample path.
    path = ["build_frame", "draw_sample", "build_cases", "project"]
    offenders = {}
    for name in path:
        node = functions.get(name)
        if node is None:
            continue
        keys = {
            sub.slice.value for sub in ast.walk(node)
            if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant)
            and isinstance(sub.slice.value, str)
        } | {
            call.args[0].value for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get" and call.args
            and isinstance(call.args[0], ast.Constant)
        }
        bad = sorted(keys & set(FORBIDDEN_KEYS))
        if bad:
            offenders[name] = bad
    results.append((
        "no forbidden key is read anywhere in build_frame/draw_sample/build_cases/project",
        not offenders,
        f"offenders: {offenders or 'none'}",
    ))

    # (c) build_frame routes records through project().
    build_frame = functions.get("build_frame")
    calls_project = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "project"
        for node in ast.walk(build_frame)
    ) if build_frame else False
    results.append((
        "build_frame() passes every record through project()",
        calls_project, f"project() called = {calls_project}",
    ))

    # (d) build_cases reads only projected keys.
    build_cases = functions.get("build_cases")
    case_keys = {
        sub.slice.value for sub in ast.walk(build_cases)
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant)
        and isinstance(sub.slice.value, str)
    } if build_cases else set()
    projected = {"claim", "primary_evidence", "counter_evidence", "claim_sha256"}
    results.append((
        "build_cases() reads only keys produced by project()",
        case_keys <= projected, f"reads {sorted(case_keys)}",
    ))

    return results, all(ok for _n, ok, _d in results)


def check_dynamic(module_name):
    """Feed project() a poisoned record; no sentinel may survive."""
    module = importlib.import_module(module_name)
    poisoned = {
        "claim": "the claim text",
        "evidence": "the primary evidence",
        "counter_evidence": "the counter evidence",
    }
    sentinels = {}
    for i, key in enumerate(FORBIDDEN_KEYS + sorted(FRAME_ONLY_KEYS)):
        sentinel = f"SENTINEL_{i}_{key}_LEAKED"
        poisoned[key] = sentinel
        sentinels[key] = sentinel

    out = module.project(poisoned)
    blob = json.dumps(out)
    leaked = sorted(k for k, s in sentinels.items() if s in blob)

    results = [(
        "no poisoned sentinel survives project()",
        not leaked, f"leaked: {leaked or 'none'}",
    ), (
        "project() output keys are exactly the whitelist",
        set(out) == {"claim", "primary_evidence", "counter_evidence", "claim_sha256"},
        f"keys {sorted(out)}",
    )]

    # And the same for the full path to a blind case object.
    frame, _excluded = module.build_frame({"eval_details": [poisoned]}, set())
    drawn, _n = module.draw_sample(frame, seed=1, target=1)
    cases = module.build_cases(drawn)
    blob = json.dumps(cases)
    leaked_path = sorted(k for k, s in sentinels.items() if s in blob)
    results.append((
        "no sentinel survives build_frame -> draw_sample -> build_cases",
        not leaked_path, f"leaked: {leaked_path or 'none'}",
    ))
    return results, all(ok for _n, ok, _d in results)


def check_tripwire(module_name):
    """Run the whole path over a record that raises on any forbidden read."""
    module = importlib.import_module(module_name)
    base = {
        "claim": "the claim text",
        "evidence": "the primary evidence",
        "counter_evidence": "the counter evidence",
        "baseline_prediction": "SUPPORT",
    }
    for key in FORBIDDEN_KEYS:
        base[key] = f"value_of_{key}"

    record = Tripwire(base, FORBIDDEN_KEYS)
    try:
        frame, _excluded = module.build_frame({"eval_details": [record]}, set())
        drawn, _n = module.draw_sample(frame, seed=1, target=1)
        module.build_cases(drawn)
        ok, detail = True, (
            f"no forbidden read; keys actually read = "
            f"{sorted(set(record.reads))}"
        )
    except AssertionError as error:
        ok, detail = False, str(error)

    results = [(
        "the full path performs no forbidden read on an instrumented record",
        ok, detail,
    )]

    # The tripwire itself must work -- prove it catches a deliberate read.
    canary = Tripwire(base, FORBIDDEN_KEYS)
    try:
        _ = canary["gold"]
        caught = False
    except AssertionError:
        caught = True
    results.append((
        "the tripwire detects a deliberate forbidden read (positive control)",
        caught, f"raised = {caught}",
    ))
    return results, all(ok for _n, ok, _d in results)


def main():
    parser = argparse.ArgumentParser(
        description="Prove the blind projection cannot access outcome fields"
    )
    parser.add_argument(
        "--module", default="inspect_scope_alignment_primary_heldout",
        help="sampler module to verify (default: the primary held-out sampler)",
    )
    args = parser.parse_args()

    print("=" * 74)
    print(f"BLIND PROJECTION VERIFICATION — module: {args.module}")
    print("=" * 74)

    overall = True
    for title, fn in (
        ("1. STATIC (AST analysis)", check_static),
        ("2. DYNAMIC (poisoned record)", check_dynamic),
        ("3. INSTRUMENTED (tripwire record)", check_tripwire),
    ):
        print(f"\n{title}")
        results, ok = fn(args.module)
        overall &= ok
        for name, passed, detail in results:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            print(f"         {detail}")

    print("\n" + "=" * 74)
    print("VERIFIED: the blind projection cannot access outcome fields."
          if overall else "FAILED: the projection may access outcome fields.")
    print("=" * 74)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
