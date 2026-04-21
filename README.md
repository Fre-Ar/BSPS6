# BSPS6

- characterize the datasets
- formulate hypothesis
- think about what we want to achieve before running experiments

2-3 hypothesis

## Characterizations
- *Spectral complexity*: Compute the angular power spectrum C_l (the distribution of signal energy across spherical harmonic degrees l). This tells you the effective bandwidth of the signal — how much high-frequency content it contains. A rapidly decaying C_l means the signal is smooth; a slow decay means high-frequency detail. You'd expect CMB to decay fast, ETOPO1 to decay slowly, with ERA5 and 360° images somewhere in between.
- *Effective bandwidth*: Derived from the power spectrum — the degree L at which, say, 95% of signal energy is captured. This gives a single number summarizing spectral complexity per dataset.
- *Isotropy*: Is the signal statistically uniform across all directions? CMB is nearly isotropic; ERA5 and ETOPO1 are strongly anisotropic (continents, poles). This matters because non-isotropic signals stress encodings differently depending on where they place frequency budget.
- *Dynamic range and spatial gradient*: Max/min signal values and average gradient magnitude. Sharp gradients indicate the presence of discontinuities or edges that NIRs struggle with.

- *Frequency Content (Spectral Power)*
- *Spatial Isotropy (Directional Uniformity)*
- *Spherical Harmonic Power Spectrum*

## Hypothesis
*H1 - Polar Singularities*: Naive angular encoding (lat/lon) performs systematically worse than all other encodings across all datasets and architectures.
- "Naive angular coordinates $(\lambda, \phi)$ will exhibit severe localized artifacts and higher RMSE at the poles due to coordinate singularities, whereas embedding inputs into 3D Cartesian space $(x,y,z)$ or using Spherical Harmonics will yield uniform error distribution across the sphere."
> Motivation: pole singularities and the longitude wrap-around discontinuity are coordinate artifacts that create regions where the network's input space is geometrically inconsistent with the signal.

*H2*: Spherical harmonic features outperform other encodings on spectrally simple signals (low effective bandwidth), but this advantage diminishes or reverses on high-frequency signals.
- "Spherical Harmonic (SH) features will be the most parameter-efficient encoding for global geophysical fields (CMB, ERA5) due to their alignment with the physics of the sphere, but will fail to reconstruct the high-frequency localized sharp edges of SUN360 images."
> Motivation: SH features are bandlimited by construction — they're the ideal encoding for smooth signals but require a prohibitively large number of features to represent high-frequency content. RFF's stochastic frequency sampling should compensate on complex signals. This hypothesis directly connects the dataset characterization to the encoding comparison, and gives you a nuanced finding rather than a blanket winner.

*H3*: The ranking of coordinate encodings is consistent across all four architectures.
> Motivation: if the encoding effect is architecture-dependent, it suggests the encoding interacts with specific inductive biases rather than being a geometry-level property. If it's consistent, you can make a stronger claim: coordinate encoding is the dominant design choice for spherical NIRs, more important than which architecture you use. This validates the paper's framing.




# Datasets Characterizations
## ETOPO1_512x1024.nc
==================================================
 DATASET CHARACTERISTICS SUMMARY 
==================================================
Dynamic Range:      16273.52 (Min: -9823.14, Max: 6450.38)
Mean Gradient:      313.27 per pixel
99th %ile Gradient: 2189.56 per pixel (Sharpness)
--------------------------------------------------
Isotropy CV:        0.6540 (Higher = more anisotropic)
--------------------------------------------------
Effective Bandwidth (L_95%): Degree 25
==================================================