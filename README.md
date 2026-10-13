# Hidden Color Contrast Violations in Mobile User Interfaces Under Color Vision Deficiency

Reproducibility repository for the CHIuXiD 2026 paper
*"Hidden Color Contrast Violations in Mobile User Interfaces Under Color Vision Deficiency."*

The study measures UI text elements that **pass** WCAG 2.1 contrast (4.5:1) under normal
vision but **fall below** the threshold under simulated Color Vision Deficiency (CVD) —
"hidden violations." Extraction uses k-means foreground/background estimation on the
**RICO** Android-UI dataset; CVD is simulated with the **Machado et al. (2009)** model;
and results are cross-validated on the denoised **CLAY** re-annotation.

## Repository contents

| Path | Description |
|------|-------------|
| `cvd_pipeline.py` | **Stage 1 — extraction.** Parses RICO view hierarchies, extracts a foreground/background color per text element (k-means, k=2), simulates the three CVD types, and computes WCAG contrast before and after. |
| `reproduce_paper.ipynb` | **Stage 2 — analysis.** Regenerates every table, figure, and statistic in the paper from the result files. Each cell prints its computed value beside the paper's reported value in `[brackets]`. |
| `results/cvd_results_random.csv` | Per-element results for the random 5,000-screen RICO sample (primary analysis, 42,285 elements). |
| `results/clay_results.csv` | Per-element results for the CLAY re-annotation (cross-dataset replication). |
| `sampled_screen_ids.txt` | The 4,381 RICO screen IDs analyzed, so the extraction can process exactly the paper's sample. |
| `extraction_labels.csv`, `extraction_labels_2.csv` | Manual labels for the general extraction-accuracy check (RICO). |
| `extraction_labels_flagged.csv` | Manual labels for flagged-violation precision (RICO). |
| `clay_extraction_labels_flagged.csv` | Manual labels for flagged-violation precision (CLAY). |
| `extraction_reasons.csv` | Reason codes for mismatched flagged elements. |

## Requirements

Python 3.9+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

(`numpy`, `matplotlib`, `scikit-learn`, `Pillow`, `daltonlens`, `jupyter`)

## Reproducing the paper's results

No raw images are needed — all tables, figures, and statistics are regenerated from the
result CSVs:

```bash
jupyter notebook reproduce_paper.ipynb   # then: Run All
```

The notebook reproduces, among others: the overall prevalence and Wilson CI (§IV-A),
extraction accuracy (§IV-B), the per-type distribution and Cochran's Q (§IV-C, Table I),
the component analysis and z-test (§IV-D, Table II), threshold proximity (§IV-E), the
severity gradient (§IV-F, Fig. 4, via `daltonlens`), model robustness (§IV-G), the CLAY
replication (§IV-H), the scatter plot (Fig. 3), and the screen-clustered bootstrap.

## Re-running the extraction from scratch (optional)

To regenerate the per-element results from raw data:

1. Download the RICO dataset from <https://interactionmining.org/rico>.
2. Place the screenshots (`.jpg`) and view-hierarchy files (`.json`) under
   `unique_uis/combined/`.
3. Run `python cvd_pipeline.py`. With `sampled_screen_ids.txt` present, it processes
   exactly the 4,381 screens analyzed in the paper and writes `results/cvd_results.csv`.

## Reproducibility

- **Analysis — exact.** `reproduce_paper.ipynb` regenerates every table, figure, and
  statistic in the paper *exactly*, from the provided result CSVs. This is the canonical
  reproduction path and requires no raw images.
- **Extraction — approximate.** Re-running `cvd_pipeline.py` on the raw RICO images
  reproduces the per-element colors and the paper's findings, but is **not guaranteed to be
  byte-identical** to the shipped CSVs. The k-means step is seeded (`random_state=0`), so it
  is deterministic for a given scikit-learn version, but different library versions or
  floating-point behaviour can shift a small number of borderline colors. The result CSVs in
  this repository are therefore the **canonical** result files.

## Data notes

- **Raw RICO screenshots/JSON are not included** — they belong to the RICO dataset and
  should be downloaded separately. This repository ships the **processed per-element
  results**, so the analysis reproduces without the raw images.
- CVD is simulated with the Machado et al. (2009) severity-parameterized matrices
  (severity 1.0 = full dichromacy for the headline rate).

## Citation

If you use this code or data, please cite:

```
L. T. Wijaya and K. Iskandar, "Hidden Color Contrast Violations in Mobile User Interfaces Under Color
Vision Deficiency," in Proc. 12th International HCI and UX Conference in Indonesia
(CHIuXiD), 2026.
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
