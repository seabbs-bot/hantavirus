## Alternative distributional assumptions for the incubation period and the
## transmission timing δ, for the explore/dist-alternatives analysis.
##
## Each variant is a separate `@model` that mirrors `joint_model` (model.jl)
## but swaps either the incubation distribution or the δ distribution.
##
## Incubation variants (must be supported on (0, ∞)):
##   * `joint_model_inc_gamma`   — Gamma(α, θ), mean/SD-style priors.
##   * `joint_model_inc_weibull` — Weibull(α, θ), mean/SD-style priors.
##
## δ variant (must support negative values):
##   * `joint_model_delta_t`     — μ_δ + σ_δ · TDist(ν), heavy-tailed.
##
## Hand-rolled Student-T logpdf (avoids the LocationScale → SpecialFunctions
## call that Enzyme can't trace through).
@inline function _logpdf_lst(x, μ, σ, ν)
    z = (x - μ) / σ
    return SpecialFunctions.loggamma((ν + 1) / 2) -
           SpecialFunctions.loggamma(ν / 2) -
           0.5 * log(π * ν) - log(σ) -
           ((ν + 1) / 2) * log(1 + z^2 / ν)
end
##
## All variants keep the rest of the joint model identical to `joint_model`:
## non-centred RW for log R(t), 1/√k NB dispersion prior, interval-censored
## T_onset/T_inf, GI > 0 reject, clamp on log_R.

# ---------------------------------------------------------------------------
# Gamma incubation
# ---------------------------------------------------------------------------
#
# Parameterisation: shape α, scale θ. Mean = αθ. Variance = αθ². We place
# priors that imply a prior-predictive mean of Inc broadly comparable to the
# current LogNormal (≈ 20 d) with a wide CV (≈ 0.5–1):
#   α ~ truncated(Normal(4.0, 2.0); lower = 0.5)
#   θ ~ truncated(Normal(5.0, 3.0); lower = 0.5)
# Prior implies mean αθ ≈ 20 d with the 95% prior on the mean roughly 5–55 d.

@model function joint_model_inc_gamma(d, edges)
    α_inc ~ truncated(Normal(4.0, 2.0); lower = 0.5)
    θ_inc ~ truncated(Normal(5.0, 3.0); lower = 0.5)
    μ_δ   ~ Normal(0.0, 5.0)
    σ_δ   ~ truncated(Normal(0.0, 1.0); lower = 0)
    phi_inv_sqrt ~ truncated(Normal(0.0, 1.0); lower = 0)
    k := 1.0 / phi_inv_sqrt^2
    σ_rw  ~ truncated(Normal(0.0, 0.2); lower = 0)

    T = typeof(α_inc)

    n_bins = length(edges) + 1
    log_R_init ~ Normal(log(1.5), 1.0)
    ε ~ Turing.filldist(Normal(zero(T), one(T)), n_bins - 1)
    log_R := vcat(log_R_init, log_R_init .+ accumulate(+, σ_rw .* ε))

    inc_dist = Gamma(α_inc, θ_inc)

    T_onset = Vector{T}(undef, d.N)
    for i in 1:d.N
        T_onset[i] ~ Uniform(d.onset_lo_day[i], d.onset_hi_day[i])
    end

    T_inf = Vector{T}(undef, d.N)
    for i in 1:d.N
        if d.source_idx[i] == 0
            T_inf[i] ~ Uniform(d.onset_lo_day[i] - 80.0, T_onset[i] - 1e-6)
            inc_i = T_onset[i] - T_inf[i]
            Turing.@addlogprob! logpdf(inc_dist, inc_i)
        else
            src = d.source_idx[i]
            T_inf[i] ~ Uniform(d.exp_lo_day[i], d.exp_hi_day[i])
            if T_inf[i] <= T_inf[src]
                Turing.@addlogprob! -Inf
            else
                inc_i  = T_onset[i] - T_inf[i]
                δ_pair = T_inf[i] - T_onset[src]
                Turing.@addlogprob! logpdf(inc_dist, inc_i)
                Turing.@addlogprob! logpdf(Normal(μ_δ, σ_δ), δ_pair)
            end
        end
        R_i = exp(clamp(log_R[which_bin(T_inf[i], edges)], -50.0, 50.0))
        d.Zobs[i] ~ NegativeBinomial(k, k / (k + R_i))
    end
end

# ---------------------------------------------------------------------------
# Weibull incubation
# ---------------------------------------------------------------------------
#
# Parameterisation: shape α, scale θ. Mean = θ·Γ(1 + 1/α). For α ≈ 2, θ ≈ 22
# the mean is ≈ 19.5 d, matching the LogNormal centre.
#   α ~ truncated(Normal(2.0, 1.0); lower = 0.5)
#   θ ~ truncated(Normal(22.0, 8.0); lower = 1.0)

@model function joint_model_inc_weibull(d, edges)
    α_inc ~ truncated(Normal(2.0, 1.0); lower = 0.5)
    θ_inc ~ truncated(Normal(22.0, 8.0); lower = 1.0)
    μ_δ   ~ Normal(0.0, 5.0)
    σ_δ   ~ truncated(Normal(0.0, 1.0); lower = 0)
    phi_inv_sqrt ~ truncated(Normal(0.0, 1.0); lower = 0)
    k := 1.0 / phi_inv_sqrt^2
    σ_rw  ~ truncated(Normal(0.0, 0.2); lower = 0)

    T = typeof(α_inc)

    n_bins = length(edges) + 1
    log_R_init ~ Normal(log(1.5), 1.0)
    ε ~ Turing.filldist(Normal(zero(T), one(T)), n_bins - 1)
    log_R := vcat(log_R_init, log_R_init .+ accumulate(+, σ_rw .* ε))

    inc_dist = Weibull(α_inc, θ_inc)

    T_onset = Vector{T}(undef, d.N)
    for i in 1:d.N
        T_onset[i] ~ Uniform(d.onset_lo_day[i], d.onset_hi_day[i])
    end

    T_inf = Vector{T}(undef, d.N)
    for i in 1:d.N
        if d.source_idx[i] == 0
            T_inf[i] ~ Uniform(d.onset_lo_day[i] - 80.0, T_onset[i] - 1e-6)
            inc_i = T_onset[i] - T_inf[i]
            Turing.@addlogprob! logpdf(inc_dist, inc_i)
        else
            src = d.source_idx[i]
            T_inf[i] ~ Uniform(d.exp_lo_day[i], d.exp_hi_day[i])
            if T_inf[i] <= T_inf[src]
                Turing.@addlogprob! -Inf
            else
                inc_i  = T_onset[i] - T_inf[i]
                δ_pair = T_inf[i] - T_onset[src]
                Turing.@addlogprob! logpdf(inc_dist, inc_i)
                Turing.@addlogprob! logpdf(Normal(μ_δ, σ_δ), δ_pair)
            end
        end
        R_i = exp(clamp(log_R[which_bin(T_inf[i], edges)], -50.0, 50.0))
        d.Zobs[i] ~ NegativeBinomial(k, k / (k + R_i))
    end
end

# ---------------------------------------------------------------------------
# Student-T δ
# ---------------------------------------------------------------------------
#
# Pre-symptomatic transmission means δ can be negative; we keep the LogNormal
# incubation (matching the headline model) and only vary the δ distribution.
# A heavier-tailed Student-T is more robust to outlying per-pair δs.
#   ν ~ Gamma(2, 5) (mean 10, shape 2) — weakly informative, allows
#   intermediate tail weight to fully Normal-like behaviour.

@model function joint_model_delta_t(d, edges)
    μ_inc ~ Normal(3.0, 0.5)
    σ_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    μ_δ   ~ Normal(0.0, 5.0)
    σ_δ   ~ truncated(Normal(0.0, 1.0); lower = 0)
    ν_δ   ~ Gamma(2.0, 5.0)
    phi_inv_sqrt ~ truncated(Normal(0.0, 1.0); lower = 0)
    k := 1.0 / phi_inv_sqrt^2
    σ_rw  ~ truncated(Normal(0.0, 0.2); lower = 0)

    T = typeof(μ_inc)

    n_bins = length(edges) + 1
    log_R_init ~ Normal(log(1.5), 1.0)
    ε ~ Turing.filldist(Normal(zero(T), one(T)), n_bins - 1)
    log_R := vcat(log_R_init, log_R_init .+ accumulate(+, σ_rw .* ε))

    inc_dist = LogNormal(μ_inc, σ_inc)

    T_onset = Vector{T}(undef, d.N)
    for i in 1:d.N
        T_onset[i] ~ Uniform(d.onset_lo_day[i], d.onset_hi_day[i])
    end

    T_inf = Vector{T}(undef, d.N)
    for i in 1:d.N
        if d.source_idx[i] == 0
            T_inf[i] ~ Uniform(d.onset_lo_day[i] - 80.0, T_onset[i] - 1e-6)
            inc_i = T_onset[i] - T_inf[i]
            Turing.@addlogprob! logpdf(inc_dist, inc_i)
        else
            src = d.source_idx[i]
            T_inf[i] ~ Uniform(d.exp_lo_day[i], d.exp_hi_day[i])
            if T_inf[i] <= T_inf[src]
                Turing.@addlogprob! -Inf
            else
                inc_i  = T_onset[i] - T_inf[i]
                δ_pair = T_inf[i] - T_onset[src]
                Turing.@addlogprob! logpdf(inc_dist, inc_i)
                Turing.@addlogprob! _logpdf_lst(δ_pair, μ_δ, σ_δ, ν_δ)
            end
        end
        R_i = exp(clamp(log_R[which_bin(T_inf[i], edges)], -50.0, 50.0))
        d.Zobs[i] ~ NegativeBinomial(k, k / (k + R_i))
    end
end
