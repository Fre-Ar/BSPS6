# BSPS6

## What do we want to achieve?
* INRs -> compressed, contiguous, differentiable signal from a discrete one
* SOTA: Euclidean domains (1D audio, 2D images, 3D shapes).
* A lot of data is inherently spherical: 
    - anything globe (weather, elevation, etc)
    - most things space (satellite imagery, astronomical surveys)
    - 360° camera panoramas from VR headsets, environment maps for HDR lighting in graphics and VFX
    - brain-surface scans in medical imaging
    - LiDAR from spinning sensors.
=> the sphere's topology introduces design choices that simply don't exist on a flat domain
* Goal: Answer "What are those choices, and which ones matter most?"
* One of these choices is "how to represent a point on a sphere as numbers?"
    - (lat,long)
    - (x,y,z)
    - spherical harmonics 
    - spherical RFF
## Choice of datasets
2 main axes: 
* bandwidth (how much small-scale detail the signal contains)
* isotropy (whether the signal's character depends on where you are on the sphere or is statistically uniform across it)
+ others: sharpness, dynamic range, and scalar vs RGB.

### CMB: Cosmic Microwave Background
stationary, isotropic, nearly gaussian random field.
### ERA5: global temperature map 
low frequency (smooth), but anisotropic 
### ETOPO1: global elevation map
mixes smooth (oceans) and sharp (continents) features at multiple spatial scales
-> literature says grid-based methods outperform INRs
### HDRI (sky): 360° panorama of a sky
low bandwidth, moderate anisotropy: RGB control case
### HDRI (urban): 360° panorama of the Shanghai skyline at sunset
sharp sky–water horizon: High bandwidth, strongly anisotropic, step-function at the horizon

## Characterizations
*Spectral complexity (C_l) & Effective bandwidth (L_95)*
Think of harmonics (sine waves), and how any 1d signal can be represented as a weighted sum of sine waves.
Spherical harmonics are those sine waves, but for 3D.
* The signal's power spectrum C_l shows how much of its total variance lives at each spherical harmonic degrees l.
* Effective bandwidth, or L_95  is the smallest degree at which we've captured 95% of that variance. Ex. A signal with L_95 = 13 (ERA5) is essentially smooth: thirteen global wave patterns account for nearly everything. 

*Isotropy*
* Variance of the signal per band of latitude.
-> Answers "Is the signal statistically uniform across all directions?"

*Spatial gradient...*
per-pixel gradient magnitude
    *dynamic range*: min & max of the spatial gradient
    *mean*: average gradient magnitude
    *sharpness*: the 99th percentile of the spatial gradient captures the sharpest 1% of edges, normalized across signals with different units by dividing by the dynamic range. 
        - High normalized P99 means the signal has locally sharp edges relative to its overall variation (coastlines, urban building outlines, etc) -> presence of edges that INRs struggle with

## Hypotheses
*H1 - Polar Singularities*: Naive angular encoding (lat/lon) performs systematically worse than all other encodings across all datasets and architectures.
- "Naive angular coordinates (lat, long) will exhibit severe localized artifacts (particularly at the ±180° longitude seam) and higher RMSE at the poles due to coordinate singularities, whereas embedding inputs into 3D Cartesian space (x,y,z) or using Spherical Harmonics will yield uniform error distribution across the sphere."
> @How to test@: after training, bin the per-pixel error by latitude and look for pole-biased residuals in the angular models that don't appear in the others.

*H2 - Spectral Math*: Spherical harmonic features outperform other encodings on spectrally simple signals (low effective bandwidth), but this advantage diminishes or reverses on high-frequency signals.
- "SH features are the frequency basis on the sphere. If a signal's L_95 fits within our cap L_max (set to 32, giving 1089 features), SH has everything it needs: the downstream network just has to learn a linear combination. If L_95 exceeds L_max, SH is physically capped, and encodings like RFF that cover a wider effective frequency range (random frequencies drawn from a broad distribution) should pull ahead."
  
> @How to test@: We expect SH to win on ERA5 (L_95 = 13) and HDRI sky (L_95 = 31), tie or fall behind on ETOPO1 (L_95 = 45), and lose clearly on CMB (L_95 = 236) and HDRI urban (L_95 = 134).

*H3 - Encoding dominance*: The ranking of coordinate encodings is consistent across all four architectures.
- "Which coordinate encoding is used for INRs learning on spherical signals matters more than which network architecture the INR has."
> @How to test@: run every (encoding × architecture × dataset) cell of the grid and ask whether the ranking of encodings is consistent across architectures and datasets, and whether it's more consistent than the ranking of architectures is across encodings and datasets. 

*H4 — Characterization predicts performance*: 3 scalars per dataset (L_95, Isotropy, normalized P99). 
- "It is possible to predict fitting performance of a INR model given spherical signal characterization"
> @How to test@: fit a small regression PSNR ≈ f(L_95, CV, P99_norm, encoding, arch) on the 80-cell grid and report which metrics are predictive for which encoding.
=> spherical extension of Vonderfecht & Liu 2024's "SIREN error prediction"


*H5 — Characterization is actionable*: L_max-matching for SH.
- "SH encoding with L_max ≈ L_95 achieves within ε dB of SH with L_max=32 on the same dataset, at significantly lower input dimension."
> @How to test@: Add one extra SH column per dataset.
=> If true, tells how to set the SH hyperparameter. If false, spectral headroom matters beyond just capturing 95% of energy.

## Data collection

*Procedure*: 
- For each cell of the grid (encoding × arch × dataset × seed), what gets logged?
- At minimum: 
  * training PSNR per epoch
  * validation PSNR per epoch
  * final test PSNR
  * per-region PSNR breakdowns (polar bands, equatorial band; per-channel for HDRI). 
- Define the test set explicitly
- uniform spherical sampling of N points, fixed seed, identical across all cells.
- Define early-stopping criterion before the fact (e.g., "patience=10 epochs on val PSNR, max 200 epochs").
- 3 seeds minimum to compute a standard error and run paired tests. 

*Statistical test per hypothesis, defined in advance*:
- H1 (polar artifacts): paired Wilcoxon signed-rank test on (polar PSNR − equatorial PSNR) for angular vs. Cartesian, across the 5 datasets × 3 seeds = 15 paired observations per arch.
- H4 (characterization predicts): leave-one-dataset-out cross-validation R² with a 95% bootstrap CI. Pre-commit to "we claim characterization is predictive iff CV R² > 0.5 with CI excluding 0."
- Variance decomposition (your H3 reformulated): two-way ANOVA with (encoding, arch) factors, report η² for each. Pre-commit to "encoding dominates iff η²(encoding) > 2 × η²(arch)."
- explicitly scope the paper to "five representative spherical signals" and never claim generalization

- Quantitative result for H2: for L_95 < 30 SH dominates by ≥3 dB; for L_95 > 60 RFF dominates; in between it's tied
- For H3: quantitative attribution to how much encoding choice matters versus architecture choice on S².
- For H4: given a new spherical signal, you can compute three cheap statistics and predict which encoding will work best, without running the model.
For H5: characterization is prescriptive, not just descriptive; you can pick a hyperparameter from a signal statistic. 

