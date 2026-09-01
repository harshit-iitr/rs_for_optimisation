# OPT 2026 submission

`main.tex` — the paper. Built against the OPT 2026 style files in this folder
(`opt2026.cls`, `jmlr.cls`, `jmlrutils.sty`), copied unmodified from
`../opt2026_style/`.

## Compiling

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**The local TeX install is missing `algorithm2e`,** which `opt2026.cls` loads
unconditionally. Overleaf and any full TeX Live have it; on this machine install
`texlive-science` or fetch the package from CTAN. Nothing in `main.tex` uses
algorithm2e, and the class was not modified to work around it.

## Conventions

- `\PH{...}` marks a placeholder and renders red. Every unfilled number, figure
  and TODO uses it. **Grep for `\PH{` before submitting; the document should
  contain none.**
- Anonymous by default (`\documentclass[anon]{opt2026}`). Switch to
  `\documentclass{opt2026}` for the camera-ready.
- Main text is limited to 6 pages excluding references and appendices. It is
  currently exactly 6, so anything added needs something removed.

## Status

Numbers already in the draft come from the Round 5 runs and are real, but are
wrapped in `\PH{}` until the arms they come from reach their planned seed count.
Still missing: baselines (Prop. 3), optimizer interaction (§7), curvature row,
per-neuron selectivity figure, second task family.

Figures in `figs/` are placeholders. Two of them (`fig_equilibrium`,
`fig_frontier`) can be generated from data already on disk.
