#!/usr/bin/env python3
"""Stouffer combination of per-scene paired McNemar p-values.

Codex review P3: when AE-3 is reported across multiple scenes, do not
pool raw command-level outcomes (assumes scene-exchangeability) and do
not naive-multiply per-scene p-values. Instead use Stouffer's combined
test on signed z-statistics, which respects per-scene weighting and
direction of effect:

  z_i  = sign(delta_i) * Phi^{-1}(1 - p_i / 2)
  Z    = sum_i (w_i * z_i) / sqrt(sum_i w_i^2)
  p    = 2 * (1 - Phi(|Z|))

Default weights w_i are sqrt of per-scene n; pass --equal-weights to
use w_i = 1.

Usage:
  python3 scripts/eval/stouffer.py \
      --inputs paper/tables/C_llm_vs_noLLM_eligible_meeting.json \
               paper/tables/C_llm_vs_noLLM_eligible_livinglab.json \
      --label "LLM vs no-LLM eligible (two-scene)" \
      --output paper/tables/C_llm_vs_noLLM_eligible_stouffer.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _ndtri(p: float) -> float:
    """Inverse standard-normal CDF using Acklam's algorithm. No SciPy
    dependency; accurate to ~1.15e-9 for p in (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
           ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path,
                    help="Per-scene mcnemar.py output JSONs to combine.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--equal-weights", action="store_true",
                    help="Use w_i=1; default is w_i=sqrt(n_i).")
    args = ap.parse_args()

    rows = []
    for path in args.inputs:
        rep = json.loads(path.read_text())
        ov = rep["overall"]
        n = ov["n"]
        if n == 0:
            continue
        # Clamp p to avoid Phi^{-1}(0) or Phi^{-1}(1).
        p = min(max(ov["p_value_exact"], 1e-12), 1 - 1e-12)
        delta = ov["delta_acc_llm_minus_baseline"]
        sign = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
        z = sign * _ndtri(1 - p / 2)
        w = 1.0 if args.equal_weights else math.sqrt(n)
        rows.append({"path": str(path), "n": n, "p_exact": ov["p_value_exact"],
                     "delta": delta, "sign": sign, "z": z, "weight": w})

    num = sum(r["weight"] * r["z"] for r in rows)
    den = math.sqrt(sum(r["weight"] ** 2 for r in rows)) if rows else 1.0
    Z = num / den if den else 0.0
    p_combined = 2.0 * (1.0 - _phi(abs(Z)))

    out = {
        "label": args.label,
        "method": "stouffer_signed_z",
        "weighting": "equal" if args.equal_weights else "sqrt_n",
        "per_scene": rows,
        "Z": Z,
        "p_value_two_sided": p_combined,
        "n_total": sum(r["n"] for r in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"  Z={Z:.3f}  p_two_sided={p_combined:.4f}  "
          f"weighting={out['weighting']}  n_total={out['n_total']}  scenes={len(rows)}")


if __name__ == "__main__":
    main()
