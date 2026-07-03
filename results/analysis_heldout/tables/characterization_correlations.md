# 2.3 characterization correlations (Spearman ρ per PE cell × feature)

**Sample size:** n = 5

**Summary:** Spearman ρ(per-dataset mean PSNR, feature) for 5 PE cells × 3 features:
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


**Notes:** n=5 datasets per correlation. Spearman ρ is rank-only — we make no linearity claim. Sign convention: positive ρ means higher feature value associates with higher PSNR. See preregistration §2.3 for the theoretical sign priors.


## Statistics

| Name | Value |
|------|-------|
| `fkan.L_95.rho` | -0.4 |
| `fkan.L_95.p_value` | 0.5046 |
| `fkan.L_95.ci` | [-1.000, 1.000] |
| `fkan.CV.rho` | 0.1 |
| `fkan.CV.p_value` | 0.8729 |
| `fkan.CV.ci` | [-1.000, 1.000] |
| `fkan.P99_norm.rho` | -0.9 |
| `fkan.P99_norm.p_value` | 0.03739 |
| `fkan.P99_norm.ci` | [-1.000, -0.111] |
| `none_angular.L_95.rho` | -0.7 |
| `none_angular.L_95.p_value` | 0.1881 |
| `none_angular.L_95.ci` | [-1.000, 0.881] |
| `none_angular.CV.rho` | 0.3 |
| `none_angular.CV.p_value` | 0.6238 |
| `none_angular.CV.ci` | [-1.000, 1.000] |
| `none_angular.P99_norm.rho` | -0.7 |
| `none_angular.P99_norm.p_value` | 0.1881 |
| `none_angular.P99_norm.ci` | [-1.000, 0.677] |
| `none_cartesian.L_95.rho` | -0.4 |
| `none_cartesian.L_95.p_value` | 0.5046 |
| `none_cartesian.L_95.ci` | [-1.000, 1.000] |
| `none_cartesian.CV.rho` | 0.1 |
| `none_cartesian.CV.p_value` | 0.8729 |
| `none_cartesian.CV.ci` | [-1.000, 1.000] |
| `none_cartesian.P99_norm.rho` | -0.9 |
| `none_cartesian.P99_norm.p_value` | 0.03739 |
| `none_cartesian.P99_norm.ci` | [-1.000, -0.111] |
| `rff.L_95.rho` | 0.5 |
| `rff.L_95.p_value` | 0.391 |
| `rff.L_95.ci` | [-1.000, 1.000] |
| `rff.CV.rho` | -0.6 |
| `rff.CV.p_value` | 0.2848 |
| `rff.CV.ci` | [-1.000, 1.000] |
| `rff.P99_norm.rho` | -0.6 |
| `rff.P99_norm.p_value` | 0.2848 |
| `rff.P99_norm.ci` | [-1.000, 1.000] |
| `sh.L_95.rho` | -0.4 |
| `sh.L_95.p_value` | 0.5046 |
| `sh.L_95.ci` | [-1.000, 1.000] |
| `sh.CV.rho` | 0.1 |
| `sh.CV.p_value` | 0.8729 |
| `sh.CV.ci` | [-1.000, 1.000] |
| `sh.P99_norm.rho` | -0.9 |
| `sh.P99_norm.p_value` | 0.03739 |
| `sh.P99_norm.ci` | [-1.000, -0.111] |
