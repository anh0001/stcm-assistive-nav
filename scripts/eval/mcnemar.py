#!/usr/bin/env python3
"""Paired McNemar test + 95% CI for AE-3 LLM ablation.

Compares STCM+LLM vs STCM no-LLM on the SAME command set, using top-1
hit vectors from `grounding_llm.py` (LLM) and `grounding.py` (template).

Reports:
  - n, b (LLM right / template wrong), c (LLM wrong / template right)
  - exact two-sided binomial p-value (preferred when b+c is small)
  - chi-square McNemar statistic w/ continuity correction (for reference)
  - paired 95% CI on accuracy delta (Wilson-style on b/n vs c/n discordants)
  - per-subset breakdown

Usage:
  python3 scripts/eval/mcnemar.py \
      --llm output/grounding_llm/meeting_grounding.json \
      --baseline output/grounding/meeting_grounding.json \
      --output paper/tables/C_meeting_mcnemar.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _binomial_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def _exact_two_sided(b: int, c: int) -> float:
    """Two-sided exact binomial test on b among n=b+c with p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    one_side = sum(_binomial_pmf(i, n, 0.5) for i in range(0, k + 1))
    p = 2.0 * one_side
    return min(p, 1.0)


def _mcnemar_chi2_cc(b: int, c: int) -> tuple[float, float]:
    """Chi-square w/ continuity correction; returns (chi2, p)."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    chi2 = max(chi2, 0.0)
    # Survival function of chi-square with 1 dof, no scipy dep:
    # p = erfc(sqrt(chi2 / 2)).
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _paired_delta_ci(b: int, c: int, n: int) -> tuple[float, float, float]:
    """95% CI on accuracy delta = (b - c) / n via normal approximation
    with paired-discordant variance. Returns (delta, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    delta = (b - c) / n
    # Variance of paired difference of indicators:
    #   Var = (b + c)/n^2 - (b - c)^2 / n^3
    var = (b + c) / (n * n) - ((b - c) ** 2) / (n ** 3)
    var = max(var, 0.0)
    half = 1.96 * math.sqrt(var)
    return delta, max(-1.0, delta - half), min(1.0, delta + half)


def _hits_by_id(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = {}
    order: list[str] = []
    for r in report.get("trials", []):
        cid = str(r.get("id"))
        rows[cid] = {
            "subset": str(r.get("subset", "unknown")),
            "top1": bool(r.get("top1")),
            "eligible": bool(r.get("eligible_grounding_given_perception", True)),
            "failure_source": str(r.get("failure_source", "")),
        }
        order.append(cid)
    return rows, order


def _summarize(b: int, c: int, n: int, llm_hits: int, base_hits: int) -> dict[str, Any]:
    p_exact = _exact_two_sided(b, c)
    chi2, p_chi2 = _mcnemar_chi2_cc(b, c)
    delta, lo, hi = _paired_delta_ci(b, c, n)
    llm_acc = llm_hits / n if n else 0.0
    base_acc = base_hits / n if n else 0.0
    llm_lo, llm_hi = _wilson_ci(llm_hits, n)
    base_lo, base_hi = _wilson_ci(base_hits, n)
    return {
        "n": n,
        "llm_top1": llm_hits,
        "baseline_top1": base_hits,
        "llm_top1_acc": llm_acc,
        "baseline_top1_acc": base_acc,
        "llm_top1_acc_ci95": [llm_lo, llm_hi],
        "baseline_top1_acc_ci95": [base_lo, base_hi],
        "discordant_b_llm_only": b,
        "discordant_c_baseline_only": c,
        "delta_acc_llm_minus_baseline": delta,
        "delta_acc_ci95": [lo, hi],
        "p_value_exact": p_exact,
        "mcnemar_chi2_cc": chi2,
        "p_value_chi2_cc": p_chi2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", required=True, type=Path,
                    help="grounding_llm.py score-phase output JSON (or any "
                         "grounding report; arm A in the pair).")
    ap.add_argument("--baseline", required=True, type=Path,
                    help="grounding.py output JSON (or any grounding report; "
                         "arm B in the pair).")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--include-subsets", default=None,
                    help="Comma-separated subset names to keep "
                         "(e.g. 'simple,disambiguation,compositional'). "
                         "Restricts both reports to these subsets before "
                         "pairing. Used for spatial-only vs functional-only "
                         "primary/secondary tables.")
    ap.add_argument("--label", default=None,
                    help="Free-form label recorded in the output JSON to "
                         "identify the partition (e.g. 'spatial_30').")
    ap.add_argument("--eligible-only", action="store_true",
                    help="Restrict pairing to commands where the GT referent "
                         "and required anchors are present in the predicted "
                         "graph in BOTH arms. Yields the grounding-given-"
                         "perception conditional comparison (Codex review P5).")
    args = ap.parse_args()

    llm_report = json.loads(args.llm.read_text())
    base_report = json.loads(args.baseline.read_text())

    llm_rows, order = _hits_by_id(llm_report)
    base_rows, _ = _hits_by_id(base_report)

    if args.include_subsets:
        wanted = {s.strip() for s in args.include_subsets.split(",") if s.strip()}
        llm_rows = {cid: r for cid, r in llm_rows.items() if r["subset"] in wanted}
        base_rows = {cid: r for cid, r in base_rows.items() if r["subset"] in wanted}
        order = [cid for cid in order if cid in llm_rows]

    if args.eligible_only:
        llm_rows = {cid: r for cid, r in llm_rows.items() if r["eligible"]}
        base_rows = {cid: r for cid, r in base_rows.items()
                     if r["eligible"] and cid in llm_rows}
        llm_rows = {cid: r for cid, r in llm_rows.items() if cid in base_rows}
        order = [cid for cid in order if cid in llm_rows]

    common_ids = [cid for cid in order if cid in base_rows]
    if len(common_ids) != len(order) or len(common_ids) != len(base_rows):
        missing_in_base = [cid for cid in order if cid not in base_rows]
        missing_in_llm = [cid for cid in base_rows if cid not in llm_rows]
        raise SystemExit(
            "Command id mismatch between paired reports.\n"
            f"  in LLM not in baseline: {missing_in_base}\n"
            f"  in baseline not in LLM: {missing_in_llm}"
        )

    overall_b = overall_c = overall_n = 0
    overall_llm_hits = overall_base_hits = 0
    by_subset: dict[str, dict[str, int]] = defaultdict(
        lambda: {"b": 0, "c": 0, "n": 0, "llm": 0, "base": 0}
    )

    for cid in common_ids:
        l = llm_rows[cid]["top1"]
        b = base_rows[cid]["top1"]
        subset = llm_rows[cid]["subset"]
        st = by_subset[subset]
        st["n"] += 1
        overall_n += 1
        st["llm"] += int(l)
        overall_llm_hits += int(l)
        st["base"] += int(b)
        overall_base_hits += int(b)
        if l and not b:
            overall_b += 1
            st["b"] += 1
        elif b and not l:
            overall_c += 1
            st["c"] += 1

    # Holm-Bonferroni adjustment across by-subset p-values to address
    # multiple-comparison concerns when reporting per-subset McNemars
    # alongside the overall test (Codex review P4).
    subset_ps = [(sub, _exact_two_sided(st["b"], st["c"]))
                 for sub, st in by_subset.items()]
    subset_ps_sorted = sorted(subset_ps, key=lambda x: x[1])
    m = len(subset_ps_sorted)
    holm_adj: dict[str, float] = {}
    running_max = 0.0
    for rank, (sub, p) in enumerate(subset_ps_sorted, start=1):
        adj = min(1.0, max(running_max, (m - rank + 1) * p))
        running_max = adj
        holm_adj[sub] = adj

    by_subset_out = {}
    for sub, st in sorted(by_subset.items()):
        s = _summarize(st["b"], st["c"], st["n"], st["llm"], st["base"])
        s["p_value_exact_holm_adj"] = holm_adj[sub]
        by_subset_out[sub] = s

    out = {
        "llm_report": str(args.llm),
        "baseline_report": str(args.baseline),
        "label": args.label,
        "include_subsets": args.include_subsets,
        "eligible_only": args.eligible_only,
        "overall": _summarize(overall_b, overall_c, overall_n,
                              overall_llm_hits, overall_base_hits),
        "by_subset": by_subset_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")
    ov = out["overall"]
    print(f"  n={ov['n']}  llm={ov['llm_top1_acc']:.3f}  "
          f"baseline={ov['baseline_top1_acc']:.3f}  "
          f"delta={ov['delta_acc_llm_minus_baseline']:+.3f}  "
          f"p_exact={ov['p_value_exact']:.4f}")


if __name__ == "__main__":
    main()
