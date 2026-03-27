# Methodology Notes

## Phase 1

Validate the source image footprint using:

- TIFF alpha mask
- ROI GML footprint

## Phase 2

Prepare model-ready RGB inputs:

- select RGB bands
- apply mask
- resample to target resolution

## Phase 3

Run and compare:

- CHMv1
- CHMv2

## Comparison focus

- visual quality
- canopy-height distributions
- min/max behavior
- edge artifacts
- usefulness for electric pole analysis



## Objective

This project evaluates CHMv1 and CHMv2 for electric pole analysis using high-resolution satellite imagery. The goal is to build a clean, reproducible pipeline for preprocessing, inference, and model comparison.

## Input data

The starting point is a 4-band GeoTIFF satellite product delivered with companion metadata files, including:

- the main product XML
- ROI footprint GML
- additional mask files
- preview products

The source raster is georeferenced and already provided in a projected CRS.

## Masking strategy

A key preprocessing decision is how to define valid image area and background.

### Preferred masking source

The preferred masking source is the TIFF alpha band.

Why:

- the source product metadata indicates that band 4 is the alpha channel
- this is a TIFF-based masking approach rather than a color-based guess
- it is directly tied to the delivered image product

### Cross-validation source

The alpha mask is cross-validated against the ROI GML footprint.

Why:

- the ROI GML provides an official footprint polygon from the delivered metadata
- agreement between alpha and ROI supports the correctness of the masking strategy

### Validation result

The TIFF alpha mask and ROI GML footprint were compared pixel by pixel.

Results:

- Pixels where masks agree: `7,354,101`
- Percent agreement: `99.9325%`
- Valid in alpha only: `1,604`
- Valid in ROI only: `3,367`

Interpretation:

- the alpha mask and ROI footprint are nearly identical
- small edge differences are expected due to polygon rasterization and boundary effects
- alpha masking is supported as the primary preprocessing mask

## Shared preprocessing for CHMv1 and CHMv2

The following steps are shared across both models:

- select bands `1, 2, 3` as RGB
- use band `4` as the valid-data mask
- preserve projected CRS and georeferencing
- preserve the same scene footprint for both models
- save intermediate and output rasters as GeoTIFFs

This ensures that both CHMv1 and CHMv2 are evaluated on the same area and imagery content.

## CHMv1 preprocessing

CHMv1 has explicit raster-format requirements in the official inference notebook.

Required or recommended setup:

- RGB imagery
- projected coordinates such as UTM or topocentric coordinates
- square pixels
- approximately `0.6 m` GSD
- `uint8` imagery
- `nodata = 255`

For this reason, CHMv1 preprocessing includes:

- creating a 3-band RGB GeoTIFF
- applying the alpha-derived valid-data mask
- setting invalid pixels to `255`
- writing `nodata = 255`
- resampling to `0.6 m`

## CHMv2 preprocessing

CHMv2 is less explicit about raster formatting in the public README and example materials.

Observed expectations:

- RGB imagery
- CHMv2 satellite normalization
- model loading via PyTorch Hub or Hugging Face
- optional tiling for large images

The public CHMv2 materials do not explicitly require:

- `0.6 m`
- `uint8`
- `nodata = 255`

For this reason, CHMv2 preprocessing is lighter:

- create a shared masked RGB input
- preserve georeferencing
- apply CHMv2 normalization during inference
- tile large rasters when needed

## Recommended comparison pipeline

The recommended controlled-comparison workflow is:

1. Start from the original 4-band GeoTIFF.
2. Use bands `1, 2, 3` as RGB.
3. Use band `4` as the valid-data mask.
4. Validate the alpha mask against the ROI GML footprint.
5. Create a masked 3-band RGB GeoTIFF.
6. Resample to `0.6 m` for CHMv1 and controlled side-by-side comparison.
7. For CHMv1, enforce `uint8` and `nodata = 255`.
8. For CHMv2, use the same masked RGB image and apply CHMv2 normalization during inference.
9. Save outputs as aligned GeoTIFFs for analysis.

## Comparison outputs

CHMv1 and CHMv2 will be compared using:

- canopy-height summary statistics
- histograms and percentile distributions
- minimum and maximum canopy-height locations
- spatial agreement and disagreement patterns
- edge artifacts and footprint behavior
- usefulness for electric pole analysis

## References

- [HighResCanopyHeight (CHMv1)](https://github.com/facebookresearch/HighResCanopyHeight)
- [DINOv3 / CHMv2](https://github.com/facebookresearch/dinov3#canopy-height-maps-v2-chmv2)
