---
description: Preflight check before JACIII revision submission (figures, numbers, acronyms, track-changes)
---

Final-mile audit for JACIII Jc26-0002 (AE-6, AE-5, AE-11..13).

Checks:
1. **Figures** — every PDF/PNG under `paper/figs/` has `dpi>=300` and in-figure font `>=10pt`. Fail if Fig. 9 not split into 9a + 9b.
2. **Table numbers** — every value in Tables A–G traceable to a script under `scripts/` or `results/`. No hand-typed numbers.
3. **Acronyms** — every abbreviation defined at first occurrence; `LIFGIF` absent; glossary Table G present.
4. **Track changes** — `paper/revision.tex` (or diff file) exists showing track-changes vs. prior submission.
5. **Contributions** — Introduction ends with exactly 4 numbered contributions matching response_to_reviewers.pdf AE-10.
6. **Length** — Introduction ≤ 2 double-spaced pages; Related Work ≤ 2.5 pages.

Report: green-only if all pass. Else enumerate gaps mapped to AE-/R- item.

Hints:
- Use `pdfinfo` / `identify` for dpi.
- `grep -i LIFGIF paper/*.tex` must return empty.
- Cross-ref Tables B/C/D against `results/eval/*.json` and `results/bench/*.json`.
