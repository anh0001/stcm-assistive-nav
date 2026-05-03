#!/usr/bin/env python3
"""Stratified (CMH-style) McNemar test on per-scene paired discordants.

Codex review P5: Stouffer combines per-scene p-values, but with only 2-3
scenes a stratified paired test on discordant counts is also informative
and arguably more transparent.

For paired binary outcomes with strata k=1..K we have per-scene
discordant counts (b_k, c_k). Under the null (no marginal difference
between arms within any scene), b_k ~ Binomial(b_k + c_k, 0.5).

This script computes:
  - sum_b, sum_c: total discordants across scenes
  - exact two-sided binomial test on (sum_b, sum_b + sum_c, 0.5)
    (the "stratified McNemar" / pooled-discordant test)
  - Mantel-Haenszel chi-square with continuity correction:
      chi2 = (|sum_k (b_k - c_k)| - 0.5)^2 / sum_k (b_k + c_k)
  - per-stratum direction agreement (sanity)

This is a sensitivity check, not a replacement for the per-scene
McNemar tests. We report all three (per-scene exact, Stouffer-combined,
stratified) and let the reader confirm consistency.

Usage:
  python3 scripts/eval/stratified_mcnemar.py \
      --inputs paper/tables/C_llm_vs_noLLM_eligible_meeting.json \
               paper/tables/C_llm_vs_noLLM_eligible_livinglab.json \
      --label "LLM vs no-LLM eligible (stratified)" \
      --output paper/tables/C_llm_vs_noLLM_eligible_stratified.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _binomial_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def _exact_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    one_side = sum(_binomial_pmf(i, n, 0.5) for i in range(0, k + 1))
    return min(2.0 * one_side, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path,
                    help="Per-scene mcnemar.py output JSONs.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    strata = []
    sum_b = 0
    sum_c = 0
    sum_d = 0  # total discordants
    sum_n = 0  # total commands
    for path in args.inputs:
        rep = json.loads(path.read_text())
        ov = rep["overall"]
        b = int(ov["discordant_b_llm_only"])
        c = int(ov["discordant_c_baseline_only"])
        n = int(ov["n"])
        delta = float(ov["delta_acc_llm_minus_baseline"])
        p_per = float(ov["p_value_exact"])
        strata.append({
            "path": str(path),
            "scene": rep.get("label") or path.stem,
            "n": n,
            "b": b,
            "c": c,
            "delta": delta,
            "p_per_scene_exact": p_per,
            "sign_delta": 1 if delta > 0 else (-1 if delta < 0 else 0),
        })
        sum_b += b
        sum_c += c
        sum_d += b + c
        sum_n += n

    p_strat_exact = _exact_two_sided(sum_b, sum_c)
    if sum_d > 0:
        diff = abs(sum_b - sum_c)
        chi2_mh = max(0.0, (diff - 0.5) ** 2 / sum_d)
        p_chi2_mh = math.erfc(math.sqrt(chi2_mh / 2.0))
    else:
        chi2_mh = 0.0
        p_chi2_mh = 1.0

    direction_agreement = (
        all(s["sign_delta"] >= 0 for s in strata) or
        all(s["sign_delta"] <= 0 for s in strata)
    )

    out = {
        "label": args.label,
        "method": "stratified_mcnemar_pooled_discordants",
        "strata": strata,
        "sum_b_llm_only": sum_b,
        "sum_c_baseline_only": sum_c,
        "n_total": sum_n,
        "n_discordant_total": sum_d,
        "p_value_exact_pooled_binomial": p_strat_exact,
        "mantel_haenszel_chi2_cc": chi2_mh,
        "p_value_chi2_cc": p_chi2_mh,
        "direction_agreement_across_strata": direction_agreement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"  sum_b={sum_b}  sum_c={sum_c}  n_total={sum_n}  "
          f"p_exact={p_strat_exact:.4f}  chi2_mh={chi2_mh:.3f} "
          f"p_chi2={p_chi2_mh:.4f}  agreement={direction_agreement}")


if __name__ == "__main__":
    main()
