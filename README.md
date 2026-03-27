# Electric Pole Analysis

This repository is the starting point for electric pole analysis using canopy height models and satellite imagery.

## Initial scope

- validate image masking from the source TIFF and delivered metadata
- prepare model-ready RGB inputs
- run CHMv1 and CHMv2
- compare CHMv1 and CHMv2 outputs

## Planned notebook order

1. `notebooks/01_mask_validation.ipynb`
2. `notebooks/02_prepare_inputs.ipynb`
3. `notebooks/03_run_chmv1.ipynb`
4. `notebooks/04_run_chmv2.ipynb`
5. `notebooks/05_compare_chmv1_vs_chmv2.ipynb`

## Recommended project layout

- `notebooks/`: exploratory and reporting notebooks
- `src/`: reusable Python helpers
- `data/raw/`: local raw inputs, not tracked in git
- `data/intermediate/`: local working files, not tracked in git
- `data/outputs/`: local outputs, not tracked in git
- `docs/`: method notes and comparison writeups
