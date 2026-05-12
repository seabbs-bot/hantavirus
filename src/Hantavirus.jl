module Hantavirus

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using CSV: CSV
using DataFrames: DataFrame, nrow, eachrow, passmissing
using Dates: Dates, Date, Day
using Distributions: Normal, LogNormal, Gamma, Weibull, TDist, LocationScale,
                     truncated, NegativeBinomial,
                     Uniform, logpdf, cdf, pdf
using MCMCChains: MCMCChains
using Plots: plot, plot!, hline!, histogram, histogram!, vline!, scatter, savefig
using Printf: @printf, @sprintf
using Random: Random
using Statistics: quantile
using Turing: Turing, @model, NUTS, MCMCThreads, sample, DynamicPPL
using ADTypes: AutoEnzyme
using Enzyme: Enzyme
import FlexiChains
import SpecialFunctions

include("data.jl")
include("model.jl")
include("model_variants.jl")
include("postprocess.jl")
include("main.jl")

export load_linelist, build_data, bin_edges_day, which_bin, bin_labels
export joint_model
export joint_model_inc_gamma, joint_model_inc_weibull, joint_model_delta_t
export diagnostics, vector_chain, summarise, save_posterior
export plot_rt, plot_delta_sense_check, plot_pairplot
export plot_prior_predictives, plot_posterior_predictions
export analyse, main

end
