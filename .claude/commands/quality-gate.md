---
description: Block JACIII submission until every AE-/R- reviewer item has evidence
---

Revision-status gate mapped to response_to_reviewers.pdf.

Args: `paper` (default) | `code` | `all`.

Checklist (green = evidence artifact exists + cross-refed in manuscript):

| Item | Artifact | Location |
|------|----------|----------|
| AE-1 / R1-1 | Table A (dataset) + Table B (nav+F1, 3 scenarios × ≥3 runs) | `results/eval/*.json` |
| AE-2 / R1-2 | Table F (SOTA capability matrix) + internal baselines in Table B | `paper/tables/sota.tex` |
| AE-3 / R1-3 | Table C (LLM ablation + McNemar p-values) | `results/eval/llm_ablation.json` |
| AE-4 / R1-4 | Table D (per-module latency p50/p95, FPS, HW) | `results/bench/runtime.json` |
| AE-5 / R2-1 | Table G glossary; LIFGIF removed | `paper/glossary.tex`, `grep -i LIFGIF` empty |
| AE-6 / R2-2 | All figs ≥300 dpi ≥10pt; Fig. 9 split 9a+9b | `paper/figs/` |
| AE-7 / R2-3 | Introduction ≤ 2 pages | `paper/sections/intro.tex` wordcount |
| AE-8 / R2-3 | Related Work ≤ 2.5 pages | `paper/sections/related.tex` wordcount |
| AE-9 / R2-4 | Table 1 rationale col + Table E sensitivity sweep | `results/eval/sensitivity.json` |
| AE-10 / R2-5 | 4-point contribution list in Introduction | `paper/sections/intro.tex` |
| AE-11..13 | Response-to-reviewers file + track-changes diff | `paper/response.tex`, `paper/revision.tex` |

Steps:
1. For each row: check artifact exists, check manuscript reference exists.
2. Output table: `[✓/✗] AE-x — artifact — reference`.
3. If any `✗`: block. Emit concrete fix action (command or agent call).
4. If all `✓`: print `READY FOR SUBMISSION`.

Never mark green on faith. Re-verify numbers from source files, not memory.
