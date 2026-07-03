# 2.3 characterization correlations (Spearman ρ per PE cell × feature)

**Sample size:** n = 5

**Summary:** Spearman ρ(per-dataset mean PSNR, feature) for 5 PE cells × 3 features:
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


**Notes:** n=5 datasets per correlation. Spearman ρ is rank-only — we make no linearity claim. Sign convention: positive ρ means higher feature value associates with higher PSNR. See preregistration §2.3 for the theoretical sign priors.


## Statistics

| Name | Value |
|------|-------|
| `fkan.L_95.rho` | -0.3 |
| `fkan.L_95.p_value` | 0.6238 |
| `fkan.L_95.ci` | [-1.000, 1.000] |
| `fkan.CV.rho` | 0 |
| `fkan.CV.p_value` | 1 |
| `fkan.CV.ci` | [-1.000, 1.000] |
| `fkan.P99_norm.rho` | -1 |
| `fkan.P99_norm.p_value` | 1.404e-24 |
| `fkan.P99_norm.ci` | [-1.000, -1.000] |
| `none_angular.L_95.rho` | -0.9 |
| `none_angular.L_95.p_value` | 0.03739 |
| `none_angular.L_95.ci` | [-1.000, -0.111] |
| `none_angular.CV.rho` | 0.4 |
| `none_angular.CV.p_value` | 0.5046 |
| `none_angular.CV.ci` | [-0.875, 1.000] |
| `none_angular.P99_norm.rho` | -0.6 |
| `none_angular.P99_norm.p_value` | 0.2848 |
| `none_angular.P99_norm.ci` | [-1.000, 1.000] |
| `none_cartesian.L_95.rho` | -0.3 |
| `none_cartesian.L_95.p_value` | 0.6238 |
| `none_cartesian.L_95.ci` | [-1.000, 1.000] |
| `none_cartesian.CV.rho` | 0 |
| `none_cartesian.CV.p_value` | 1 |
| `none_cartesian.CV.ci` | [-1.000, 1.000] |
| `none_cartesian.P99_norm.rho` | -1 |
| `none_cartesian.P99_norm.p_value` | 1.404e-24 |
| `none_cartesian.P99_norm.ci` | [-1.000, -1.000] |
| `rff.L_95.rho` | 0.8 |
| `rff.L_95.p_value` | 0.1041 |
| `rff.L_95.ci` | [0.111, 1.000] |
| `rff.CV.rho` | -0.7 |
| `rff.CV.p_value` | 0.1881 |
| `rff.CV.ci` | [-1.000, 0.875] |
| `rff.P99_norm.rho` | -0.2 |
| `rff.P99_norm.p_value` | 0.7471 |
| `rff.P99_norm.ci` | [-1.000, 1.000] |
| `sh.L_95.rho` | -0.8 |
| `sh.L_95.p_value` | 0.1041 |
| `sh.L_95.ci` | [-1.000, -0.111] |
| `sh.CV.rho` | 0.3 |
| `sh.CV.p_value` | 0.6238 |
| `sh.CV.ci` | [-1.000, 1.000] |
| `sh.P99_norm.rho` | -0.7 |
| `sh.P99_norm.p_value` | 0.1881 |
| `sh.P99_norm.ci` | [-1.000, 1.000] |
