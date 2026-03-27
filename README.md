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


## Preprocessing Plan

This project compares CHMv1 and CHMv2 for electric pole analysis using high-resolution satellite imagery.

### preprocessing_shared

The following preprocessing steps are shared across both models:

- Use only the 3 RGB bands from the source GeoTIFF.
- Use TIFF alpha masking and/or ROI GML footprint masking to define valid pixels.
- Preserve georeferencing and projected CRS information.
- Keep the same study area and valid footprint for both models.
- Save outputs as GeoTIFFs for map-based comparison.

Why:

- Both models are designed to work with RGB imagery.
- TIFF-based masking is more reliable than inferring invalid pixels from RGB color alone.
- Keeping a shared footprint and map alignment makes CHMv1 vs CHMv2 comparison fair and reproducible.

Proof:

- CHMv1 official notebook states it runs on an input RGB image: [run_chm_model.ipynb](https://raw.githubusercontent.com/facebookresearch/HighResCanopyHeight/main/notebooks/run_chm_model.ipynb)
- CHMv2 official loading examples also use RGB imagery: [DINOv3 README](https://github.com/facebookresearch/dinov3#canopy-height-maps-v2-chmv2)
- The source product metadata indicates band 4 is the alpha channel and provides an ROI footprint polygon.

### preprocessing_chmv1_only

The following preprocessing steps are explicitly required or strongly recommended for CHMv1:

- Resample the image to approximately `0.6 m` ground sample distance.
- Ensure pixels are square.
- Use `uint8` input imagery.
- Set invalid input pixels to `255` and use `nodata = 255`.
- Use projected coordinates such as UTM or topocentric coordinates.
- Run inference using tiled/windowed processing for larger rasters.

Why:

- These are directly stated in the official CHMv1 notebook and are part of the expected inference setup.

Proof:

- The official notebook states:
  - input RGB image
  - topocentric coordinates recommended
  - pixel GSD of `0.6 m`
  - `Uint8`
  - `nodata = 255`
- Source: [run_chm_model.ipynb](https://raw.githubusercontent.com/facebookresearch/HighResCanopyHeight/main/notebooks/run_chm_model.ipynb)

### preprocessing_chmv2_only

The following preprocessing steps are specific to CHMv2:

- Apply CHMv2 satellite normalization during inference.
- Use either the CHMv2 PyTorch Hub loading path or the Hugging Face `AutoImageProcessor`.
- Tile larger rasters manually if full-scene inference is too large for memory.
- Keep mask handling outside the model and apply it at the raster-processing stage.

Why:

- CHMv2 official examples focus on RGB image inference and model normalization.
- The CHMv2 materials do not explicitly require `0.6 m`, `uint8`, or `nodata = 255` in the same way CHMv1 does.

Proof:

- CHMv2 official model loading examples are described here: [DINOv3 README](https://github.com/facebookresearch/dinov3#canopy-height-maps-v2-chmv2)
- The Hugging Face example uses `AutoImageProcessor` and `AutoModelForDepthEstimation`.
- The CHMv2 tiled local notebook uses satellite normalization values:
  - mean `(0.420, 0.411, 0.296)`
  - std `(0.213, 0.156, 0.143)`

### recommended_pipeline_for_this_project

The recommended workflow for this repository is:

1. Start from the original 4-band GeoTIFF.
2. Use bands `1, 2, 3` as RGB.
3. Use band `4` as the valid-data mask.
4. Cross-check the alpha mask against the ROI GML footprint.
5. Create a masked 3-band RGB GeoTIFF as the shared model input.
6. Resample to `0.6 m` for CHMv1 and for fair side-by-side comparison.
7. For CHMv1, enforce `uint8` and `nodata = 255`.
8. For CHMv2, use the same masked RGB image but apply CHMv2 normalization at inference time.
9. Save outputs as GeoTIFFs and compare:
   - canopy-height distributions
   - min and max canopy-height locations
   - edge artifacts
   - visual usefulness for electric pole analysis

Why this approach:

- It keeps the comparison controlled and reproducible.
- It follows CHMv1’s published raster assumptions.
- It uses TIFF-based masking rather than RGB-based guessing.
- It allows CHMv2 to be evaluated on the same footprint and scene content.

### Mask validation note

The source TIFF alpha mask and the ROI GML footprint were cross-validated and showed very strong agreement.

Results from notebook validation:

- Pixels where masks agree: `7,354,101`
- Percent agreement: `99.9325%`
- Valid in alpha only: `1,604`
- Valid in ROI only: `3,367`

Interpretation:

- The TIFF alpha mask and ROI GML footprint are nearly identical.
- This supports using alpha-based masking as the primary masking strategy for preprocessing.
- The ROI GML remains a strong metadata-based validation source.
