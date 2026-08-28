# Activation Geometry & Out-of-Distribution Generalization LaTeX Report

This folder contains a unified, detailed LaTeX report summarizing the theoretical foundations, local experiments, and peer findings of our work on **Activation Norm Regularization / Radial Suppression**.

## Contents

- **`report.tex`**: The primary LaTeX document containing:
  - Theoretical framework and derivations for Proposition 1 (Edge of Stability), Proposition 2 (Lagrangian Riemannian Flow relaxation), and Proposition 3 (Antagonistic Gradients/Spectral Collapse).
  - Details and results for local continual learning experiments (Permuted MNIST v2, Split CIFAR-10, Rotating MNIST).
  - Compilation of peer findings (Harshit's CelebA/Camelyon17 spurious correlation tests, Aditya's BERT/ViT sample efficiency sweeps, and Sarthak's Split CIFAR-100 Leaky ReLU and ablation runs).
  - Tables summarizing performance, dead neurons, and representation rank.
- **`figs/`**: Contains all plots generated from our runs and compiled from peer folders.
- **`report.pdf`**: The compiled 10-page PDF report.

## Compilation Instructions

If you modify `report.tex`, you can compile it using `pdflatex`:

```bash
pdflatex report.tex
pdflatex report.tex
```

*(Note: compiling twice ensures all cross-references, lists of figures, and the table of contents are populated correctly.)*
