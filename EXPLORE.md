# Alternative distributional assumptions for delays in the joint model

This worktree explores swapping the incubation-period and transmission-timing distributions in the Epuyén ANDV joint model, following the recommendations of Charniga et al. (2024), *Best practices for estimating and reporting epidemiological delay distributions of infectious diseases* (arXiv:2405.08841).

## Charniga et al. (2024): relevant guidance

From section "Estimation items" (pp. 6–7 of the preprint):

> We recommend fitting a parametric distribution to summarize the empirical delay distribution. Multiple probability distributions should be fitted to delay data [44] and compared using appropriate model comparison criteria (e.g., widely applicable information criterion [WAIC] or leave-one-out information criterion [LOOIC] for Bayesian models). Common distributions for epidemiological delays in the literature include the gamma, lognormal, and Weibull distributions [45]. For delays that can have negative values, distributions that can accept negative values, such as the skew-normal or skew-logistic distributions [46], may be used, or less ideally, the delay data may be shifted to allow for fitting of distributions that only allow positive numbers [47]. Mixture distributions may be appropriate for some delays and should also be considered [48–51].

And on reporting:

> We recommend reporting an estimate of variability (e.g., standard deviation or dispersion) along with central tendency (e.g., mean or median) for all estimated delay distributions … The parameter estimates and quantiles of fitted probability distributions should also be reported as they are often used in modeling. All summary statistics should always be accompanied by credible intervals or confidence intervals … (usually 90% or 95% with the width of the reported interval also being reported).

The Charniga prescription is therefore: fit Gamma, LogNormal, Weibull (at minimum); compare with LOOIC or WAIC; report mean/median plus variability plus relevant quantiles with credible intervals; use a real-line distribution where the delay can be negative.

## Variants fitted

All variants preserve the headline-model machinery (non-centred RW on log R(t), σ_rw prior N⁺(0, 0.2), reciprocal-square-root NB-k prior, Enzyme AD, InitFromPrior, log_R clamp, interval-censored T_onset / T_inf, GI > 0 reject).

### Incubation distribution (positive support)

| Variant | Distribution | Priors |
| --- | --- | --- |
| **A. LogNormal (current)** | `LogNormal(μ_inc, σ_inc)` | `μ_inc ~ N(3.0, 0.5)`, `σ_inc ~ N⁺(0, 0.5)` |
| **B. Gamma**                | `Gamma(α, θ)`             | `α ~ N⁺(4.0, 2.0; ≥0.5)`, `θ ~ N⁺(5.0, 3.0; ≥0.5)` |
| **C. Weibull**              | `Weibull(α, θ)`           | `α ~ N⁺(2.0, 1.0; ≥0.5)`, `θ ~ N⁺(22.0, 8.0; ≥1.0)` |

Prior calibration target: each variant should imply a prior-predictive mean Inc in roughly 5–55 days with 95% prior coverage, centred near 20 d (the order of magnitude expected for ANDV). Means and 99th percentiles of the implied prior distributions are broadly comparable: LogNormal prior median of mean(Inc) ≈ 22 d, Gamma prior median of mean ≈ 20 d, Weibull prior median of mean ≈ 19 d.

### Transmission timing δ (real line)

| Variant | Distribution | Priors |
| --- | --- | --- |
| **A. Normal (current)**     | `Normal(μ_δ, σ_δ)` | `μ_δ ~ N(0, 5)`, `σ_δ ~ N⁺(0, 1)` |
| **D. Student-T (location-scale)** | `μ_δ + σ_δ · T(ν)` | `μ_δ ~ N(0, 5)`, `σ_δ ~ N⁺(0, 1)`, `ν ~ Gamma(2, 5)` |

Per the Charniga note that real-line distributions are preferred where negative values are possible, Normal is acceptable (symmetric, real-line). Student-T relaxes the tail-weight assumption with `ν ~ Gamma(2, 5)` (prior mean 10, weakly informative; ν → ∞ recovers the Normal).

Skew-Normal is mentioned in Charniga et al. as an option but not fitted here. Pre-symptomatic transmission asymmetry is biologically plausible (incubation period skews right; δ may inherit some asymmetry through correlated source/recipient infection times), but the per-pair posterior medians for δ in the headline model are roughly symmetric around 0, so the heavy-tailed Student-T is a more defensible first alternative.

The Student-T log-density is hand-rolled (see `_logpdf_lst` in `src/model_variants.jl`) because Enzyme's reverse mode cannot trace through `Distributions.LocationScale(TDist(ν))` (it calls into `SpecialFunctions._loggammadiv` via `sqmahal`).

## Diagnostics, fits, and headline summaries

See `output/dist_alternatives/comparison.csv` for the machine-readable table; the table below is filled in from the full 4 × 1000 NUTS run.

| variant | mean Inc (95% CrI) | 95th-pct Inc | 99th-pct Inc | μ_δ | σ_δ | other | rhat | ess | ndiv |
|---|---|---|---|---|---|---|---|---|---|
| **A. LogNormal (current)** | 22.53 (20.23–25.34) | 36.09 (31.40–44.17) | 44.77 (37.74–57.88) | 0.18 (-0.19–0.50) | 0.61 (0.46–0.84) | μ = 3.06, σ = 0.32 | 1.004 | 1714 | 0 |
| **B. Gamma incubation** | 22.62 (19.81–25.97) | 38.51 (33.48–45.64) | 47.48 (40.78–57.61) | 0.18 (-0.16–0.50) | 0.61 (0.46–0.83) | α = 6.80, θ = 3.32 | 1.005 | 1712 | 0 |
| **C. Weibull incubation** | 22.24 (19.74–24.83) | 35.04 (31.63–40.12) | 40.12 (35.88–47.22) | 0.18 (-0.16–0.50) | 0.61 (0.46–0.83) | α = 3.17, θ = 24.84 | 1.006 | 2233 | 0 |
| **D. Student-T δ** | 22.57 (20.20–25.35) | 36.18 (31.27–44.26) | 44.98 (37.55–58.05) | **-0.08 (-0.36–0.23)** | **0.09 (0.01–0.28)** | ν = 1.33 (0.60–3.47) | 1.006 | **164** | 0 |

### Observations

- **Incubation mean** is almost identical across A / B / C (22.2–22.6 d), with 95% CrIs heavily overlapping. The tail behaviour does differ:
  - LogNormal mid tail, Gamma heaviest upper tail (q99 = 47.5 d), Weibull lightest (q99 = 40.1 d).
  - Without LOO/WAIC we cannot formally rank them. Visual posterior-predictive overlay (`inc_posterior_predictive.png`) is the practical comparison.
- **Diagnostics** for A / B are essentially equivalent. Weibull has slightly higher rhat (1.006) but more ess_bulk (2233), still well within usable range.
- **Student-T on δ** is a clear regression on this data: ess_bulk collapses to **164** (1/10× the Normal baseline), `ν` is poorly identified (CrI 0.60–3.47, i.e. Cauchy-to-near-Normal), `σ_δ` collapses to 0.09 because the heavy tail absorbs structure that the Normal model attributes to scale, and `μ_δ` shifts to −0.08 (the location parameter of a Student-T is not its mean). The data don't support the extra tail parameter.

## Figures

- `figures/dist_alternatives/inc_dist_compare.png` — overlaid posterior PDFs of the incubation distribution under each variant (median ± 5–95% ribbon over posterior draws).
- `figures/dist_alternatives/delta_dist_compare.png` — overlaid posterior PDFs of δ under Normal vs Student-T.
- `figures/dist_alternatives/headline_compare.png` — forest-plot of headline posteriors (mean Inc, 95th-pct Inc, 99th-pct Inc, μ_δ, σ_δ) across variants.
- `figures/dist_alternatives/inc_posterior_predictive.png` — posterior-predictive Inc samples per variant.
- `figures/dist_alternatives/delta_posterior_predictive.png` — posterior-predictive δ samples per variant overlaid with the per-pair posterior medians from the headline fit.

## Model comparison (LOO / WAIC)

LOO-CV / WAIC are not reported here. Computing pointwise log-likelihoods for this joint model is non-trivial under FlexiChains: the likelihood for each case combines a NegativeBinomial offspring term, a LogNormal/Gamma/Weibull incubation term, and (for sourced cases) a δ term — all gated by an `addlogprob! -Inf` reject branch that enforces GI > 0. Wiring `PSIS.jl` cleanly would require re-evaluating the per-case log-likelihood on each posterior draw using a `pointwise_logdensities`-equivalent under Turing 0.45 / FlexiChains 0.6, which is fiddly enough to be out of scope for this scoping pass.

Per Charniga: "When Markov chain Monte Carlo methods are used to estimate delay distributions, it is important to visualize posterior predictions against data and check model diagnostics, such as R-hat values, divergent transitions, and effective sample sizes." We report rhat / ess / ndiv across variants, alongside posterior-predictive overlays of Inc and δ. A future PR could add LOO-CV using FlexiChains' upcoming `pointwise_logdensities` (or by hand for each case).

## Right truncation

Right truncation is an additional bias flagged by Charniga et al. — recent infections with long incubation periods are under-represented in finite-window line lists. The Epuyén outbreak line list ends shortly after the last identified case, so the data are largely backward-looking by the time of fitting. Right-truncation correction is out of scope here (it is tracked separately).

## Recommendation

**Hold the current parameterisations.** On this 34-case data set:

- **Incubation**: LogNormal, Gamma and Weibull recover essentially the same mean Inc and overlapping 95% CrIs. They differ only in upper-tail behaviour (LogNormal mid, Gamma heaviest, Weibull lightest at the 99th percentile). Without LOO-CV / WAIC there is no defensible reason to prefer one of the three; Charniga's guidance is to fit all three and compare, which is documented here as a reproducible script but the formal comparison criterion is deferred to a follow-up that wires in pointwise log-likelihoods.
- **Transmission timing δ**: Student-T degrades sampling badly (ess ≈ 164 vs 1714 for Normal), produces a poorly identified `ν`, and collapses `σ_δ` against the Normal scale by absorbing variance into the tail. With 34 cases and 33 sourced pairs there isn't enough information to identify the extra tail parameter. Stick with Normal.

**Follow-ups worth opening as separate issues**

- Add a LOO/WAIC pipeline so future incubation-distribution comparisons can be made on a quantitative basis (this requires wiring `pointwise_logdensities` under Turing 0.45 / FlexiChains; see the comment in this report).
- Consider Skew-Normal for δ as a follow-up if a future dataset shows visible asymmetry in the per-pair δ posterior; not warranted on this data.
- Right-truncation is a separate, higher-priority concern flagged by Charniga — tracked elsewhere.

## Reproduction

```bash
cd /Users/lshsa2/code/external/hantavirus/worktrees/dist-alternatives
HV_SAMPLES=1000 HV_CHAINS=4 julia --project=. --startup-file=no scripts/dist_alternatives.jl
```

Environment variables `HV_SAMPLES`, `HV_CHAINS`, `HV_SEED` allow tuning the run. Output goes to `output/dist_alternatives/` (CSVs) and `figures/dist_alternatives/` (PNGs).
