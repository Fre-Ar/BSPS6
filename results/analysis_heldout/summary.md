# Analysis summary

**Primary metric:** `held_out_psnr` (Held-out PSNR).

| Analysis | Sample n |
|----------|----------|
| variance_decomposition | 75 |
| polar_penalty_contrast | 15 |
| characterization_correlations | 5 |
| sh_lmax_ablation | 15 |

## Per-analysis details

### variance_decomposition — 2.1 variance decomposition (PE vs activation vs dataset)

η²: pe=0.283 (CI 0.143–0.479), activation=0.106 (CI 0.018–0.259), dataset=0.282 (CI 0.147–0.553).

### polar_penalty_contrast — 2.2 polar-penalty contrast (none_angular vs none_cartesian)

median Δ(none_angular − none_cartesian) = -1.367 dB (95% bootstrap CI -1.789–-0.025; n=15). Wilcoxon W = 26.00, p = 0.9760 (diagnostic only).

### characterization_correlations — 2.3 characterization correlations (Spearman ρ per PE cell × feature)

Spearman ρ(per-dataset mean PSNR, feature) for 5 PE cells × 3 features:
             fkan /      L_95: ρ=-0.400 (CI -1.000–+1.000, p=0.505)
             fkan /        CV: ρ=+0.100 (CI -1.000–+1.000, p=0.873)
             fkan /  P99_norm: ρ=-0.900 (CI -1.000–-0.111, p=0.037)
     none_angular /      L_95: ρ=-0.700 (CI -1.000–+0.881, p=0.188)
     none_angular /        CV: ρ=+0.300 (CI -1.000–+1.000, p=0.624)
     none_angular /  P99_norm: ρ=-0.700 (CI -1.000–+0.677, p=0.188)
   none_cartesian /      L_95: ρ=-0.400 (CI -1.000–+1.000, p=0.505)
   none_cartesian /        CV: ρ=+0.100 (CI -1.000–+1.000, p=0.873)
   none_cartesian /  P99_norm: ρ=-0.900 (CI -1.000–-0.111, p=0.037)
              rff /      L_95: ρ=+0.500 (CI -1.000–+1.000, p=0.391)
              rff /        CV: ρ=-0.600 (CI -1.000–+1.000, p=0.285)
              rff /  P99_norm: ρ=-0.600 (CI -1.000–+1.000, p=0.285)
               sh /      L_95: ρ=-0.400 (CI -1.000–+1.000, p=0.505)
               sh /        CV: ρ=+0.100 (CI -1.000–+1.000, p=0.873)
               sh /  P99_norm: ρ=-0.900 (CI -1.000–-0.111, p=0.037)

*Notes:* n=5 datasets per correlation. Spearman ρ is rank-only — we make no linearity claim. Sign convention: positive ρ means higher feature value associates with higher PSNR. See preregistration §2.3 for the theoretical sign priors.

### sh_lmax_ablation — 2.4 SH L_max ablation (post- and pre-saturation regimes)

Post-saturation Δ_match (L_max=⌈L_95⌉ − L_max=32): median=0.093 dB (CI -3.307–0.566, n=6). Pre-saturation Δ_LMax (L_max=32 − L_max=16): median=0.460 dB (CI -0.373–0.608, n=9).

*Notes:* Wilcoxon p-values are diagnostics, not pass/fail criteria. At n=6 (post-saturation) the smallest achievable two-sided p is ~0.031; at n=9 (pre-saturation) it is ~0.004.
