module QUnfoldExperiments

using
    CSV,
    DataFrames,
    Discretizers,
    Distributions,
    HDF5,
    H5Zblosc,
    LinearAlgebra,
    QUnfold,
    Random,
    Statistics,
    StatsBase,
    Printf
import ScikitLearnBase

const PYLOCK = Ref{ReentrantLock}()
const A_EFF = Ref{Vector{Float64}}()
const BIN_CENTERS = Ref{Vector{Float64}}()
const BIN_EDGES = Ref{Vector{Float64}}()
const FEATURES = Ref{Vector{Symbol}}()
const FACT_X_TRN = Ref{Matrix{Float32}}()
const FACT_Y_TRN = Ref{Vector{Int32}}()
const FACT_X_REM = Ref{Matrix{Float32}}()
const FACT_Y_REM = Ref{Vector{Int32}}()
const P_TRN = Ref{Vector{Float64}}()

function __init__()
    PYLOCK[] = ReentrantLock()

    # read acceptance factors
    fact_dir = "$(dirname(@__DIR__))/data/fact"
    df_acceptance = CSV.read("$(fact_dir)/acceptance.csv", DataFrame)
    A_EFF[] = df_acceptance[2:end-1, :a_eff]
    BIN_CENTERS[] = df_acceptance[2:end-1, :bin_center]
    BIN_EDGES[] = disallowmissing(vcat(
        df_acceptance[2:end-1,:e_min],
        df_acceptance[end-1,:e_max]
    ))
    ld = LinearDiscretizer(log10.(BIN_EDGES[]))

    # # alternative: bins by Max Nöthe
    # BIN_CENTERS[] = 10 .^ collect(2.8:0.2:4.4)
    # BIN_EDGES[] = 10 .^ collect(2.7:0.2:4.5)

    # read training data
    df_trn = read_gamma("$(fact_dir)/gamma_train.hdf5")
    FEATURES[] = setdiff(propertynames(df_trn), [:log10_energy])
    FACT_X_TRN[] = Matrix{Float32}(df_trn[:, FEATURES[]])
    FACT_Y_TRN[] = Int32.(encode(ld, df_trn[!, :log10_energy]))
    P_TRN[] = [ sum(FACT_Y_TRN[] .== i) / length(FACT_Y_TRN[]) for i in 1:length(BIN_CENTERS[]) ]

    # read remaining test and validation data
    df_tst = read_gamma("$(fact_dir)/gamma_test.hdf5")
    FACT_X_REM[] = Matrix{Float32}(df_tst[:, FEATURES[]])
    FACT_Y_REM[] = Int32.(encode(ld, df_tst[!, :log10_energy]))
end

# read a DataFrame from a DL2 HDF5 file of simulated gammas
function read_gamma(path::AbstractString)
    df = DataFrame()

    # read each HDF5 groups that astro-particle physicists read according to
    # https://github.com/fact-project/open_crab_sample_analysis/blob/f40c4fab57a90ee589ec98f5fe3fdf38e93958bf/configs/aict.yaml#L30
    features = [
        :size,
        :width,
        :length,
        :skewness_trans,
        :skewness_long,
        :concentration_cog,
        :concentration_core,
        :concentration_one_pixel,
        :concentration_two_pixel,
        :leakage1,
        :leakage2,
        :num_islands,
        :num_pixel_in_shower,
        :photoncharge_shower_mean,
        :photoncharge_shower_variance,
        :photoncharge_shower_max,
    ]
    @info "Reading $(path)"
    h5open(path, "r") do file
        for f in [
                :corsika_event_header_total_energy, # the target variable
                :cog_x, # needed only for feature generation; will be removed
                :cog_y,
                features...,
                ]
            df[!, f] = read(file, "events/$(f)")
        end
    end

    # generate additional features, according to
    # https://github.com/fact-project/open_crab_sample_analysis/blob/f40c4fab57a90ee589ec98f5fe3fdf38e93958bf/configs/aict.yaml#L50
    df[!, :log_size] = log.(df[!, :size])
    df[!, :area] = df[!, :width] .* df[!, :length] .* π
    df[!, :size_area] = df[!, :size] ./ df[!, :area]
    df[!, :cog_r] = sqrt.(df[!, :cog_x].^2 + df[!, :cog_y].^2)

    # the label needs to be transformed for log10 plots
    df[!, :log10_energy] = log10.(df[!, :corsika_event_header_total_energy])

    # convert the features to Float32, to find non-finite elements
    df = df[!, [:log10_energy, :log_size, :area, :size_area, :cog_r, features...]]
    for column in names(df)
        df[!, column] = convert.(Float32, df[!, column])
    end
    return filter(row -> all([ isfinite(cell) for cell in row ]), df)
end

acceptance_factors() = 1 ./ A_EFF[]
bin_centers() = BIN_CENTERS[]
bin_edges() = BIN_EDGES[]

pylock(f::Function) = Base.lock(f, PYLOCK[]) # thread safety for PyCall

mutable struct CachedClassifier
    classifier::Any
    last_X::Any
    last_predict::Vector{Int}
    last_predict_proba::Matrix{Float64}
    only_apply::Bool
    properties::Dict{Symbol,Any}
    classes::Vector{Int32}
    cold_cache_warnings::Bool
    CachedClassifier(c::Any; only_apply::Bool=false) =
        new(c, nothing, Int[], Matrix{Float64}(undef, 0, 0), only_apply, Dict{Symbol,Any}(), Int32[], false)
end

Base.hasproperty(c::CachedClassifier, x::Symbol) =
    x ∈ fieldnames(CachedClassifier) || x ∈ keys(getfield(c, :properties))
Base.getproperty(c::CachedClassifier, x::Symbol) =
    x ∈ fieldnames(CachedClassifier) ? getfield(c, x) : getfield(c, :properties)[x]
ScikitLearnBase.fit!(c::CachedClassifier, X::Any, y::AbstractVector{T}) where {T<:Integer} = pylock() do
    ScikitLearnBase.fit!(getfield(c, :classifier), X, y)
    setfield!(c, :properties, Dict(x => getproperty(getfield(c, :classifier), x) for x ∈ propertynames(getfield(c, :classifier))))
    setfield!(c, :classes, ScikitLearnBase.get_classes(getfield(c, :classifier)))
    return c
end
function ScikitLearnBase.predict(c::CachedClassifier, X::Any)
    if getfield(c, :only_apply)
        throw(ArgumentError("predict is only supported if only_apply is false"))
    elseif getfield(c, :last_X) != X
        if getfield(c, :cold_cache_warnings)
            @warn "Calling predict(c, X) before _cache!(c, X) is called"
        end
        _cache!(c, X)
    end
    return getfield(c, :last_predict)
end
function ScikitLearnBase.predict_proba(c::CachedClassifier, X::Any)
    if getfield(c, :only_apply)
        throw(ArgumentError("predict_proba is only supported if only_apply is false"))
    elseif getfield(c, :last_X) != X
        if getfield(c, :cold_cache_warnings)
            @warn "Calling predict_proba(c, X) before _cache!(c, X) is called"
        end
        _cache!(c, X)
    end
    return getfield(c, :last_predict_proba)
end
function QUnfold._apply_tree(c::CachedClassifier, X::Any)
    if !getfield(c, :only_apply)
        throw(ArgumentError("_apply_tree is only supported if only_apply is true"))
    elseif getfield(c, :last_X) != X
        if getfield(c, :cold_cache_warnings)
            @warn "Calling _apply_tree(c, X) before _cache!(c, X) is called"
        end
        _cache!(c, X)
    end
    return getfield(c, :last_predict) # hack: c.classifier.apply(X) is stored in c.last_predict
end
_cache!(c::CachedClassifier, X::Any) = pylock() do
        if getfield(c, :only_apply)
            setfield!(c, :last_predict, getfield(c, :classifier).apply(X)) # hack
        else
            pXY = ScikitLearnBase.predict_proba(getfield(c, :classifier), X)
            setfield!(c, :last_predict_proba, pXY)
            setfield!(c, :last_predict, [ last(idx.I) for idx ∈ argmax(pXY, dims=2)[:] ])
        end
        setfield!(c, :last_X, X)
    end
ScikitLearnBase.clone(c::CachedClassifier) = pylock() do
    CachedClassifier(ScikitLearnBase.clone(getfield(c, :classifier)); only_apply=getfield(c, :only_apply))
end
ScikitLearnBase.get_classes(c::CachedClassifier) = getfield(c, :classes)


"""
    fact_data([rng, ]N_trn=120000) -> (X_trn, y_trn, X_val, y_val, X_tst, y_tst)

Return a training set `(X_trn, y_trn)` of size `N_trn`, a validation pool `(X_val, y_val)`,
and a testing pool `(X_tst, y_tst)` of the FACT data.
"""
fact_data(N_trn::Int=120000) = fact_data(Random.GLOBAL_RNG, N_trn)
function fact_data(rng::AbstractRNG, N_trn::Int=120000)
    i_trn = randperm(length(FACT_Y_TRN[]))[1:N_trn]
    i_rem = randperm(length(FACT_Y_REM[]))
    N_val = floor(Int, length(i_rem) / 2) # number of validation samples
    i_val = i_rem[1:N_val]
    i_tst = i_rem[(N_val+1):end]
    return (
        FACT_X_TRN[][i_trn,:], FACT_Y_TRN[][i_trn], # training set (X_trn, y_trn)
        FACT_X_REM[][i_val,:], FACT_Y_REM[][i_val], # validation pool (X_val, y_val)
        FACT_X_REM[][i_tst,:], FACT_Y_REM[][i_tst], # testing pool (X_tst, y_tst)
    )
end

"""
    crab_data(csv_path="\$(dirname(@__DIR__))/data/fact/crab.csv") -> (X_q, X_b)

Return the observed data `X_q` and the background measurement `X_b` of FACT's
Crab data set.
"""
function crab_data(csv_path::String="$(dirname(@__DIR__))/data/fact/crab.csv")
    df_crab = DataFrames.disallowmissing!(CSV.read(csv_path, DataFrame))
    X = Matrix{Float32}(df_crab[:, FEATURES[]])
    is_on = Bool.(df_crab[!, :is_on])
    X[is_on,:], X[.!(is_on),:] # only "on" and "off" samples are present in this file
end

# a meaningful exception for subsample_indices
struct ExhaustedClassException <: Exception
    label::Int64 # the label that is exhausted
    desired::Int64
    available::Int64
end
Base.showerror(io::IO, e::ExhaustedClassException) = print(io,
    "ExhaustedClassException: Cannot sample $(e.desired) instances of class $(e.label)",
    " (only $(e.available) available)")

"""
    subsample_indices([rng,] N, p, y)

Subsample `N` indices of labels `y` with prevalences `p`.
"""
subsample_indices(
        N :: Int,
        p :: AbstractVector{T1},
        y :: AbstractVector{T2};
        kwargs...
        ) where {T1<:Real, T2<:Integer} =
    subsample_indices(Random.GLOBAL_RNG, N, p, y; kwargs...)

function subsample_indices(
        rng :: AbstractRNG,
        N :: Int,
        p :: AbstractVector{T1},
        y :: AbstractVector{T2};
        n_classes :: Int = length(unique(y)),
        allow_duplicates :: Bool = true,
        Np_min :: Int = 1,
        ) where {T1<:Real, T2<:Integer}
    if length(p) != n_classes
        throw(ArgumentError("length(p) != n_classes"))
    end
    Np = round_Np(rng, N, p; Np_min=Np_min)
    i = randperm(rng, length(y)) # random order after shuffling
    j = vcat(map(1:n_classes) do c
        i_c = (1:length(y))[y[i].==c]
        if Np[c] <= length(i_c)
            i_c[1:Np[c]] # the return value of this map operation
        elseif allow_duplicates
            @debug "Have to repeat $(ceil(Int, Np[c] / length(i_c))) times"
            i_c = repeat(i_c, ceil(Int, Np[c] / length(i_c)))
            i_c[1:Np[c]] # take from a repetition
        else
            throw(ExhaustedClassException(c, Np[c], length(i_c)))
        end
    end...) # indices of the shuffled sub-sample
    return i[j]
end

"""
    magic_crab_flux(x=BIN_CENTERS)

Compute the Crab nebula flux in `GeV⋅cm²⋅s` for a vector `x` of energy values
that are given in `GeV`. This parametrization is by Aleksić et al. (2015).
"""
magic_crab_flux(x::Union{Float64,Vector{Float64}}=BIN_CENTERS[]) =
    @. 3.23e-10 * (x/1e3)^(-2.47 - 0.24 * log10(x/1e3))

"""
    round_Np([rng, ]N, p; Np_min=1)

Round `N * p` such that `sum(N*p) == N` and `minimum(N*p) >= Np_min`. We use this
rounding to determine the number of samples to draw according to `N` and `p`.
"""
round_Np(N::Int, p::Vector{Float64}; kwargs...) =
    round_Np(Random.GLOBAL_RNG, N, p; kwargs...)

function round_Np(rng::AbstractRNG, N::Int, p::Vector{Float64}; Np_min::Int=1)
    Np = max.(round.(Int, N * p), Np_min)
    while N != sum(Np)
        ϵ = N - sum(Np)
        if ϵ > 0 # are additional draws needed?
            Np[StatsBase.sample(rng, 1:length(p), Weights(max.(p, 1/N)), ϵ)] .+= 1
        elseif ϵ < 0 # are less draws needed?
            c = findall(Np .> Np_min)
            Np[StatsBase.sample(rng, c, Weights(max.(p[c], 1/N)), -ϵ)] .-= 1
        end
    end # rarely needs more than one iteration
    return Np
end

# sample acceptance-corrected probabilities from Poisson distributions
sample_poisson(N, m) = [ sample_poisson(N) for _ in 1:m ]
function sample_poisson(N)
    p = magic_crab_flux() .* A_EFF[]
    λ = p * N ./ sum(p) # Poisson rates for N events in total
    random_sample = [ rand(Poisson(λ_i)) for λ_i in λ ]
    return round_Np(N, random_sample ./ sum(random_sample)) ./ N
end

sample_npp_crab(N, m) = [ sample_npp_crab(N) for _ in 1:m ]
function sample_npp_crab(N)
    p = magic_crab_flux() .* A_EFF[]
    return round_Np(N, p ./ sum(p)) ./ N
end

sample_npp_simulation(N, m) = [ sample_npp_simulation(N) for _ in 1:m ]
sample_npp_simulation(N) = round_Np(N, P_TRN[]) ./ N

sample_app(N, m) = [ sample_app(N) for _ in 1:m ]
sample_app(N) = round_Np(N, rand(Dirichlet(ones(length(BIN_CENTERS[]))))) ./ N

function sample_app_oq(N, m=10000, keep=.2)
    app = sample_app(N, ceil(Int, m/keep))
    c = [ curvature(log10.(x)) for x in app ]
    i = sortperm(c)[1:m]
    return app[i]
end

"""
    to_log10_spectrum_density(N, p)

Compute a logarithmic acceptance-corrected spectrum from a prevalence vector `p`. The
spectrum is, again, normalized to a probability density.
"""
function to_log10_spectrum_density(N::Int, p::AbstractVector{T}) where T<:Number
    q = p ./ A_EFF[]
    q = log10.(1 .+ q ./ sum(q) .* (N-length(q)))
    return q ./ sum(q)
end

"""
    curvature(p)

Compute the curvature `1/2 * (T * p)^2` of the probability density `p`.
"""
function curvature(p::AbstractVector{T}) where T<:Number
    t = LinearAlgebra.diagm( # Tikhonov matrix
        -1 => fill(-1, length(BIN_CENTERS[])-1),
        0 => fill(2, length(BIN_CENTERS[])),
        1 => fill(-1, length(BIN_CENTERS[])-1)
    )[2:(length(BIN_CENTERS[])-1), :]
    return (t * p)' * (t * p) / 2 # 1/2 (Tp)^2
end

"""
    nmd(a, b) = mdpa(a, b) / (length(a) - 1)

Compute the Normalized Match Distance (NMD) [sakai2021evaluating], a variant of the Earth
Mover's Distance [rubner1998metric] which is normalized by the number of classes.
"""
nmd(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    mdpa(a, b) / (length(a) - 1)

"""
    mdpa(a, b)

Minimum Distance of Pair Assignments (MDPA) [cha2002measuring] for ordinal pdfs `a` and `b`.
The MDPA is a special case of the Earth Mover's Distance [rubner1998metric] that can be
computed efficiently.
"""
function mdpa(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number
    # __check_distance_arguments(a, b)
    prefixsum = 0.0 # algorithm 1 in [cha2002measuring]
    distance  = 0.0
    for i in 1:length(a)
        prefixsum += a[i] - b[i]
        distance  += abs(prefixsum)
    end
    return distance / sum(a) # the normalization is a fix to the original MDPA
end

# format statistics of curvatures and divergences
format_statistics(x) = [ @sprintf("%.5f", x_i)[2:end] for x_i ∈ vcat(quantile(x, [.05, .25, .5, .75, .95]), mean(x)) ]

# TeX table export
export_table(output_path, df, columns="ll$(repeat("r", size(df, 2) - 2))") =
    open(output_path, "w") do io
        println(io, "\\begin{tabular}{$(columns)}")
        println(io, "  \\toprule")
        println(io, "    ", join(names(df), " & "), " \\\\") # header
        println(io, "  \\midrule")
        for r in eachrow(df)
            println(io, "    ", join(r, " & "), " \\\\")
        end
        println(io, "  \\bottomrule")
        println(io, "\\end{tabular}")
    end

end # module
