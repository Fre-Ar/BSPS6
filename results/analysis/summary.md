# Analysis summary

**Primary metric:** `reconstruction_psnr` (Reconstruction PSNR).

| Analysis | Sample n |
|----------|----------|
| variance_decomposition | 75 |
| polar_penalty_contrast | 15 |
| characterization_correlations | 5 |
| sh_lmax_ablation | 15 |

## Per-analysis details

### variance_decomposition — 2.1 variance decomposition (PE vs activation vs dataset)

η²: pe=0.326 (CI 0.192–0.525), activation=0.023 (CI 0.002–0.142), dataset=0.136 (CI 0.059–0.352).

### polar_penalty_contrast — 2.2 polar-penalty contrast (none_angular vs none_cartesian)

median Δ(none_angular − none_cartesian) = 0.017 dB (95% bootstrap CI -1.361–1.413; n=15). Wilcoxon W = 71.00, p = 0.2807 (diagnostic only).

### characterization_correlations — 2.3 characterization correlations (Spearman ρ per PE cell × feature)

Spearman ρ(per-dataset mean PSNR, feature) for 5 PE cells × 3 features:
             fkan /      L_95: ρ=-0.300 (CI -1.000–+1.000, p=0.624)
             fkan /        CV: ρ=+0.000 (CI -1.000–+1.000, p=1.000)
             fkan /  P99_norm: ρ=-1.000 (CI -1.000–-1.000, p=0.000)
     none_angular /      L_95: ρ=-0.900 (CI -1.000–-0.111, p=0.037)
     none_angular /        CV: ρ=+0.400 (CI -0.875–+1.000, p=0.505)
     none_angular /  P99_norm: ρ=-0.600 (CI -1.000–+1.000, p=0.285)
   none_cartesian /      L_95: ρ=-0.300 (CI -1.000–+1.000, p=0.624)
   none_cartesian /        CV: ρ=+0.000 (CI -1.000–+1.000, p=1.000)
   none_cartesian /  P99_norm: ρ=-1.000 (CI -1.000–-1.000, p=0.000)
              rff /      L_95: ρ=+0.800 (CI +0.111–+1.000, p=0.104)
              rff /        CV: ρ=-0.700 (CI -1.000–+0.875, p=0.188)
              rff /  P99_norm: ρ=-0.200 (CI -1.000–+1.000, p=0.747)
               sh /      L_95: ρ=-0.800 (CI -1.000–-0.111, p=0.104)
               sh /        CV: ρ=+0.300 (CI -1.000–+1.000, p=0.624)
               sh /  P99_norm: ρ=-0.700 (CI -1.000–+1.000, p=0.188)

*Notes:* n=5 datasets per correlation. Spearman ρ is rank-only — we make no linearity claim. Sign convention: positive ρ means higher feature value associates with higher PSNR. See preregistration §2.3 for the theoretical sign priors.

### sh_lmax_ablation — 2.4 SH L_max ablation (post- and pre-saturation regimes)

Post-saturation Δ_match (L_max=⌈L_95⌉ − L_max=32): median=-1.767 dB (CI -10.948–0.101, n=6). Pre-saturation Δ_LMax (L_max=32 − L_max=16): median=4.649 dB (CI -0.655–7.026, n=9).

*Notes:* Wilcoxon p-values are diagnostics, not pass/fail criteria. At n=6 (post-saturation) the smallest achievable two-sided p is ~0.031; at n=9 (pre-saturation) it is ~0.004.
