## Driver for the explore/dist-alternatives analysis. Fits the headline
## LogNormal joint model alongside Gamma- and Weibull-incubation variants
## and a Student-T δ variant, then produces a side-by-side comparison.
##
## Usage:
##   julia --project=. scripts/dist_alternatives.jl
##
## Implementation note: this script intentionally does *not* `using Hantavirus`
## because the package's `(@main)` entry point auto-invokes `analyse()` under
## Julia 1.12 (which would run the full headline analysis on every load).
## Instead we wire up an isolated module that includes only the bits we need.

module HantaDist

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using CSV: CSV
using DataFrames: DataFrame, nrow, eachrow, passmissing
using Dates: Dates, Date, Day
using Distributions: Normal, LogNormal, Gamma, Weibull, TDist, LocationScale,
                     truncated, NegativeBinomial,
                     Uniform, logpdf, cdf, pdf
using MCMCChains: MCMCChains
using Plots: plot, plot!, hline!, histogram, histogram!, vline!, scatter,
             savefig, scatter!
using Printf: @printf, @sprintf
using Random: Random
using Statistics: quantile, mean, std
using Turing: Turing, @model, NUTS, MCMCThreads, sample, DynamicPPL
using ADTypes: AutoEnzyme
using Enzyme: Enzyme
import FlexiChains
import SpecialFunctions

const ROOT_DIR = abspath(joinpath(@__DIR__, ".."))
const SRC      = joinpath(ROOT_DIR, "src")

# data.jl uses pkgdir(@__MODULE__) which returns nothing inside this
# submodule. Pre-define the path constants and inject them into the
# loaded data.jl by reading + substituting before evaluating.
let
    txt = read(joinpath(SRC, "data.jl"), String)
    txt = replace(txt,
        "joinpath(pkgdir(@__MODULE__), \"data\", \"linelist.csv\")" =>
            "joinpath(raw\"$ROOT_DIR\", \"data\", \"linelist.csv\")",
        "joinpath(pkgdir(@__MODULE__), \"output\")" =>
            "joinpath(raw\"$ROOT_DIR\", \"output\")",
        "joinpath(pkgdir(@__MODULE__), \"figures\")" =>
            "joinpath(raw\"$ROOT_DIR\", \"figures\")",
    )
    include_string(@__MODULE__, txt, joinpath(SRC, "data.jl"))
end
include(joinpath(SRC, "model.jl"))
include(joinpath(SRC, "model_variants.jl"))
include(joinpath(SRC, "postprocess.jl"))

end  # module HantaDist

using .HantaDist
using Plots: plot, plot!, savefig, scatter!, scatter, histogram, histogram!,
             vline!, hline!
using Statistics: quantile, mean, std
using Distributions: LogNormal, Gamma, Weibull, Normal, LocationScale,
                     TDist, pdf, cdf, logpdf, quantile as dquantile
using Random: Random, MersenneTwister
using Turing: Turing, NUTS, MCMCThreads, sample, DynamicPPL
using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using Printf: @printf, @sprintf
using CSV: CSV
using DataFrames: DataFrame

const ROOT       = joinpath(@__DIR__, "..")
const FIG_DIR    = joinpath(ROOT, "figures", "dist_alternatives")
const OUT_DIR    = joinpath(ROOT, "output", "dist_alternatives")
const N_SAMPLES  = parse(Int, get(ENV, "HV_SAMPLES", "1000"))
const N_CHAINS   = parse(Int, get(ENV, "HV_CHAINS",  "4"))
const SEED       = parse(Int, get(ENV, "HV_SEED",    "20260512"))

mkpath(FIG_DIR); mkpath(OUT_DIR)

Random.seed!(SEED)
ll    = HantaDist.load_linelist()
d     = HantaDist.build_data(ll)
edges = HantaDist.bin_edges_day(d.t0)
@info "Loaded line list" n_cases=d.N n_sources=sum(>(0), d.source_idx)

const ADTYPE = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))

function fit_variant(name, model)
    @info "Fitting $name" samples=N_SAMPLES chains=N_CHAINS
    t0 = time()
    chn = sample(
        model,
        NUTS(0.95; adtype = ADTYPE), MCMCThreads(), N_SAMPLES, N_CHAINS;
        initial_params = fill(DynamicPPL.InitFromPrior(), N_CHAINS),
        progress = false,
    )
    elapsed = time() - t0
    diag = HantaDist.diagnostics(chn)
    @info "$name fit done" rhat=diag.rhat ess=diag.ess ndiv=diag.ndiv seconds=elapsed
    return chn, diag, elapsed
end

# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------
# Each variant produces draws of (Inc median, Inc 95th, Inc 99th, μ_δ, σ_δ),
# alongside diagnostics. `inc_quantiles(chn)` returns Vector{NTuple} per draw.

vec1(chn, s) = vec(collect(chn[s]))

function summarise_inc_quantiles(inc_dist_fn, chn)
    mean_inc = Float64[]; q95 = Float64[]; q99 = Float64[]; sd_inc = Float64[]
    for dist in inc_dist_fn(chn)
        push!(mean_inc, mean(dist))
        push!(sd_inc,   std(dist))
        push!(q95,      dquantile(dist, 0.95))
        push!(q99,      dquantile(dist, 0.99))
    end
    return (; mean_inc, sd_inc, q95, q99)
end

function inc_dists_lognormal(chn)
    μ = vec1(chn, :μ_inc); σ = vec1(chn, :σ_inc)
    return [LogNormal(μ[i], σ[i]) for i in eachindex(μ)]
end

function inc_dists_gamma(chn)
    α = vec1(chn, :α_inc); θ = vec1(chn, :θ_inc)
    return [Gamma(α[i], θ[i]) for i in eachindex(α)]
end

function inc_dists_weibull(chn)
    α = vec1(chn, :α_inc); θ = vec1(chn, :θ_inc)
    return [Weibull(α[i], θ[i]) for i in eachindex(α)]
end

# ---------------------------------------------------------------------------
# Run all variants
# ---------------------------------------------------------------------------

models = [
    ("LogNormal (current)", HantaDist.joint_model(d, edges),              inc_dists_lognormal),
    ("Gamma incubation",    HantaDist.joint_model_inc_gamma(d, edges),    inc_dists_gamma),
    ("Weibull incubation",  HantaDist.joint_model_inc_weibull(d, edges),  inc_dists_weibull),
    ("Student-T δ",         HantaDist.joint_model_delta_t(d, edges),      inc_dists_lognormal),
]

results = Dict{String,Any}()

for (name, model, inc_fn) in models
    chn, diag, secs = fit_variant(name, model)
    inc_summ = summarise_inc_quantiles(inc_fn, chn)
    μ_δ = vec1(chn, :μ_δ); σ_δ = vec1(chn, :σ_δ)
    results[name] = (; chn, diag, secs, inc_summ, μ_δ, σ_δ, inc_fn)
end

# Save a row-per-variant CSV with the headline summaries.
qci3(x) = (quantile(x, 0.025), quantile(x, 0.5), quantile(x, 0.975))

rows = DataFrame(
    variant   = String[],
    mean_inc  = String[], q95_inc = String[], q99_inc = String[],
    mu_delta  = String[], sigma_delta = String[],
    extra     = String[],
    rhat      = Float64[], ess = Float64[], ndiv = Int[], secs = Float64[],
)

fmt(t) = @sprintf("%.2f (%.2f–%.2f)", t[2], t[1], t[3])
fmt3(t) = @sprintf("%.3f (%.3f–%.3f)", t[2], t[1], t[3])

for (name, _, _) in models
    r = results[name]
    extra = ""
    if name == "Student-T δ"
        ν = vec1(r.chn, :ν_δ)
        extra = "ν = " * fmt(qci3(ν))
    elseif name == "Gamma incubation"
        α = vec1(r.chn, :α_inc); θ = vec1(r.chn, :θ_inc)
        extra = @sprintf("α = %.2f, θ = %.2f", quantile(α, 0.5), quantile(θ, 0.5))
    elseif name == "Weibull incubation"
        α = vec1(r.chn, :α_inc); θ = vec1(r.chn, :θ_inc)
        extra = @sprintf("α = %.2f, θ = %.2f", quantile(α, 0.5), quantile(θ, 0.5))
    elseif name == "LogNormal (current)"
        μ = vec1(r.chn, :μ_inc); σ = vec1(r.chn, :σ_inc)
        extra = @sprintf("μ = %.2f, σ = %.2f", quantile(μ, 0.5), quantile(σ, 0.5))
    end
    push!(rows, (
        name,
        fmt(qci3(r.inc_summ.mean_inc)),
        fmt(qci3(r.inc_summ.q95)),
        fmt(qci3(r.inc_summ.q99)),
        fmt(qci3(r.μ_δ)),
        fmt(qci3(r.σ_δ)),
        extra,
        r.diag.rhat, r.diag.ess, r.diag.ndiv, r.secs,
    ))
end

CSV.write(joinpath(OUT_DIR, "comparison.csv"), rows)
println("\nComparison summary:")
show(stdout, "text/plain", rows)
println()

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# 1. Incubation distribution overlay: posterior median PDF + 90% ribbon
xs_inc = range(0.01, 70.0; length = 300)
function pdf_band(dist_fn, chn; xs = xs_inc, qs = (0.05, 0.5, 0.95))
    dists = dist_fn(chn)
    pdfs = [pdf.(d, xs) for d in dists]
    lo = [quantile([p[j] for p in pdfs], qs[1]) for j in eachindex(xs)]
    md = [quantile([p[j] for p in pdfs], qs[2]) for j in eachindex(xs)]
    hi = [quantile([p[j] for p in pdfs], qs[3]) for j in eachindex(xs)]
    return lo, md, hi
end

colours = Dict(
    "LogNormal (current)" => :steelblue,
    "Gamma incubation"    => :darkorange,
    "Weibull incubation"  => :seagreen,
    "Student-T δ"         => :purple,
)

plt_inc = plot(; xlabel = "incubation period (days)",
                 ylabel = "density",
                 title  = "Incubation: posterior PDFs (median, 5–95% ribbon)",
                 size   = (900, 500))
for (name, _, inc_fn) in models
    name == "Student-T δ" && continue  # same Inc dist as LogNormal
    r = results[name]
    lo, md, hi = pdf_band(r.inc_fn, r.chn)
    plot!(plt_inc, xs_inc, md; ribbon = (md .- lo, hi .- md),
          label = name, linecolor = colours[name],
          fillcolor = colours[name], fillalpha = 0.15, linewidth = 2)
end
savefig(plt_inc, joinpath(FIG_DIR, "inc_dist_compare.png"))

# 2. δ posterior densities: Normal vs Student-T.
xs_δ = range(-8.0, 8.0; length = 300)
function delta_pdfs(name)
    r = results[name]
    μ = r.μ_δ; σ = r.σ_δ
    if name == "Student-T δ"
        ν = vec1(r.chn, :ν_δ)
        return [LocationScale(μ[i], σ[i], TDist(ν[i])) for i in eachindex(μ)]
    else
        return [Normal(μ[i], σ[i]) for i in eachindex(μ)]
    end
end

plt_δ = plot(; xlabel = "δ (days from source onset)", ylabel = "density",
               title = "δ posterior: Normal vs Student-T",
               size = (900, 500))
for name in ("LogNormal (current)", "Student-T δ")
    dists = delta_pdfs(name)
    pdfs = [pdf.(d, xs_δ) for d in dists]
    lo = [quantile([p[j] for p in pdfs], 0.05) for j in eachindex(xs_δ)]
    md = [quantile([p[j] for p in pdfs], 0.5)  for j in eachindex(xs_δ)]
    hi = [quantile([p[j] for p in pdfs], 0.95) for j in eachindex(xs_δ)]
    plot!(plt_δ, xs_δ, md; ribbon = (md .- lo, hi .- md),
          label = name, linecolor = colours[name],
          fillcolor = colours[name], fillalpha = 0.15, linewidth = 2)
end
savefig(plt_δ, joinpath(FIG_DIR, "delta_dist_compare.png"))

# 3. Forest plot of headline posteriors.
labels = ["mean Inc", "95th-pct Inc", "99th-pct Inc", "μ_δ", "σ_δ"]
function variant_values(r)
    return [
        qci3(r.inc_summ.mean_inc),
        qci3(r.inc_summ.q95),
        qci3(r.inc_summ.q99),
        qci3(r.μ_δ),
        qci3(r.σ_δ),
    ]
end

plt_forest = plot(; layout = (1, length(labels)),
                    size = (1500, 400),
                    plot_title = "Headline posteriors across distributional variants")
for (j, lbl) in enumerate(labels)
    sub = plt_forest[j]
    plot!(sub; xlabel = lbl, yticks = false, framestyle = :semi,
              legend = (j == 1 ? :topright : false), title = "")
    for (i, (name, _, _)) in enumerate(models)
        r = results[name]
        vals = variant_values(r)[j]
        scatter!(sub, [vals[2]], [i]; markersize = 6, color = colours[name],
                 markerstrokewidth = 0, label = (j == 1 ? name : ""))
        plot!(sub, [vals[1], vals[3]], [i, i]; color = colours[name],
              linewidth = 2, label = "")
    end
    plot!(sub; ylims = (0.5, length(models) + 0.5))
end
savefig(plt_forest, joinpath(FIG_DIR, "headline_compare.png"))

# 4. Posterior predictive check on δ: each variant simulates δ draws from
# the fitted population distribution and compares with the per-pair posterior
# medians extracted from the chain.
function predictive_δ(name; n_per_draw = 50, rng = MersenneTwister(123))
    r = results[name]
    μ = r.μ_δ; σ = r.σ_δ
    out = Float64[]
    if name == "Student-T δ"
        ν = vec1(r.chn, :ν_δ)
        for i in eachindex(μ)
            for _ in 1:n_per_draw
                push!(out, μ[i] + σ[i] * rand(rng, TDist(ν[i])))
            end
        end
    else
        for i in eachindex(μ)
            for _ in 1:n_per_draw
                push!(out, rand(rng, Normal(μ[i], σ[i])))
            end
        end
    end
    return out
end

# Observed per-pair δ medians (use the LogNormal chain — they're nearly
# identical across variants for this exploratory check).
let
    r = results["LogNormal (current)"]
    t_inf   = HantaDist.vector_chain(r.chn, :T_inf)
    t_onset = HantaDist.vector_chain(r.chn, :T_onset)
    obs_δ = Float64[]
    for i in 1:d.N
        src = d.source_idx[i]
        src == 0 && continue
        push!(obs_δ, quantile(t_inf[i] .- t_onset[src], 0.5))
    end
    plt_pp = plot(; xlabel = "δ (days)", ylabel = "density",
                    title  = "δ posterior predictive vs observed per-pair medians",
                    size   = (900, 500), xlims = (-10, 10))
    for name in ("LogNormal (current)", "Student-T δ")
        sims = predictive_δ(name)
        sims = clamp.(sims, -50, 50)
        histogram!(plt_pp, sims; bins = range(-10, 10; length = 80),
                   normalize = :pdf, alpha = 0.4, color = colours[name],
                   label = "predictive — " * name)
    end
    histogram!(plt_pp, obs_δ; bins = 15, normalize = :pdf, alpha = 0.7,
               color = :black, label = "observed per-pair medians")
    savefig(plt_pp, joinpath(FIG_DIR, "delta_posterior_predictive.png"))
end

# 5. Posterior predictive check on Inc: simulate incubation draws from the
# fitted distribution and overlay across variants.
let
    plt_pp = plot(; xlabel = "incubation period (days)", ylabel = "density",
                    title = "Inc posterior predictive across variants",
                    size  = (900, 500), xlims = (0, 70))
    rng = MersenneTwister(42)
    for (name, _, inc_fn) in models
        name == "Student-T δ" && continue
        r = results[name]
        dists = inc_fn(r.chn)
        sims = Float64[]
        for dst in dists
            for _ in 1:50
                push!(sims, rand(rng, dst))
            end
        end
        sims = filter(x -> 0 <= x <= 80, sims)
        histogram!(plt_pp, sims; bins = range(0, 70; length = 80),
                   normalize = :pdf, alpha = 0.4,
                   color = colours[name], label = "predictive — " * name)
    end
    savefig(plt_pp, joinpath(FIG_DIR, "inc_posterior_predictive.png"))
end

@info "All figures written" dir=FIG_DIR
@info "Comparison CSV"      path=joinpath(OUT_DIR, "comparison.csv")

# Save lightweight per-variant posterior summaries to CSV for reproducibility.
for (name, _, _) in models
    r = results[name]
    short = replace(lowercase(name), r"[^a-z0-9]+" => "_")
    df = DataFrame(
        mean_inc = r.inc_summ.mean_inc,
        sd_inc   = r.inc_summ.sd_inc,
        q95_inc  = r.inc_summ.q95,
        q99_inc  = r.inc_summ.q99,
        mu_delta = r.μ_δ,
        sigma_delta = r.σ_δ,
    )
    CSV.write(joinpath(OUT_DIR, "posterior_$(short).csv"), df)
end
