# Public Nightly Engineering Smoke Test

Completed in the build environment:

- Python compilation: PASS.
- Existing ChemProcessRAG unit tests: `2 passed`.
- CrystalCV physical-ID canonicalization: PASS for known duplicate examples such as `FRC3_231013_0.2mmh` and `A_230528_0.2mmh`.
- Historical V5.2.2 CrystalCV split audit: 223 file representations collapse to 126 canonical physical IDs under the new grouping heuristic, demonstrating that duplicate representation removal is active.
- CrystalCV fixture preparation: 8/8 physical experiments prepared.
- End-to-end residual forecast smoke: five horizons ran through LastValue, LinearTrend, Ridge, MLP, GRU and TCPT and wrote JSON/CSV.
- Context/RAG audit fallback: correctly returns `AUDIT_ONLY` rather than inventing context when no explicit mapping is present.
- HeinSight YOLO split discovery: tested on a fabricated YOLO directory with grouped frame names; official held-out images remain reserved for final evaluation.

These are engineering tests only, not scientific dataset results.
