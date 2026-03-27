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


## Preprocessing Summary

This project compares CHMv1 and CHMv2 for electric pole analysis using high-resolution satellite imagery.

### Shared preprocessing

- Use bands `1, 2, 3` as RGB.
- Use TIFF alpha masking and ROI GML footprint masking to define valid pixels.
- Preserve projected CRS and georeferencing.
- Keep the same footprint and study area for both models.
- Save outputs as GeoTIFFs for direct comparison.

### CHMv1-specific preprocessing

- Resample imagery to approximately `0.6 m`.
- Ensure pixels are square.
- Use `uint8` input.
- Set invalid pixels to `255` and use `nodata = 255`.
- Use projected coordinates such as UTM.
- Run tiled/windowed inference for larger rasters.

Source: [HighResCanopyHeight](https://github.com/facebookresearch/HighResCanopyHeight)

### CHMv2-specific preprocessing

- Use RGB imagery with CHMv2 satellite normalization during inference.
- Use the official CHMv2 loading path through PyTorch Hub or Hugging Face.
- Tile larger rasters manually if needed.
- Handle masking at the raster-processing stage rather than relying on model internals.

Source: [DINOv3 / CHMv2](https://github.com/facebookresearch/dinov3#canopy-height-maps-v2-chmv2)

### Recommended pipeline for this project

1. Start from the original 4-band GeoTIFF.
2. Use bands `1, 2, 3` as RGB.
3. Use band `4` as the valid-data mask.
4. Cross-check the alpha mask against the ROI GML footprint.
5. Create a masked 3-band RGB GeoTIFF as the shared model input.
6. Resample to `0.6 m` for CHMv1 and controlled comparison.
7. Apply CHMv1-specific formatting (`uint8`, `nodata = 255`) where required.
8. Apply CHMv2 normalization during CHMv2 inference.
9. Compare CHMv1 and CHMv2 outputs using aligned GeoTIFF products.

### Mask validation result

The TIFF alpha mask and ROI GML footprint showed very strong agreement:

- Agreement: `99.9325%`
- Valid in alpha only: `1,604`
- Valid in ROI only: `3,367`

This supports using alpha-based masking as the primary preprocessing strategy.
