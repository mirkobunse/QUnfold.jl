module QUnfoldExperiments

using
    CSV,
    DataFrames,
    Discretizers,
    Distributions,
    LinearAlgebra,
    Random,
    Statistics,
    StatsBase,
    Printf

const A_EFF = Ref{Vector{Float64}}()
const BIN_CENTERS = Ref{Vector{Float64}}()
const BIN_EDGES = Ref{Vector{Float64}}()
const P_TRN = Ref{Vector{Float64}}()

function __init__()
    fact_dir = "$(dirname(@__DIR__))/data/fact"
    df_acceptance = CSV.read("$(fact_dir)/acceptance.csv", DataFrame)
    A_EFF[] = df_acceptance[2:end-1, :a_eff]
    BIN_CENTERS[] = df_acceptance[2:end-1, :bin_center]
    BIN_EDGES[] = disallowmissing(vcat(
        df_acceptance[2:end-1,:e_min],
        df_acceptance[end-1,:e_max]
    ))

    # # Bins by Max Nöthe
    # BIN_CENTERS[] = 10 .^ collect(2.8:0.2:4.4)
    # BIN_EDGES[] = 10 .^ collect(2.7:0.2:4.5)

    df_data = DataFrames.disallowmissing!(CSV.read("$(fact_dir)/fact_wobble.csv", DataFrame))
    y = encode( # labels of the simulated data
        LinearDiscretizer(log10.(BIN_EDGES[])),
        df_data[!, :log10_energy]
    )
    P_TRN[] = [ sum(y .== i) / length(y) for i in 1:length(BIN_CENTERS[]) ]
end

"""
    magic_crab_flux(x=BIN_CENTERS)

Compute the Crab nebula flux in `GeV⋅cm²⋅s` for a vector `x` of energy values
that are given in `GeV`. This parametrization is by Aleksíc et al. (2015).
"""
magic_crab_flux(x::Vector{Float64}=BIN_CENTERS[]) =
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
format_statistics(x) = [ @sprintf("%.5f", x)[2:end] for x ∈ quantile(x, [.05, .25, .5, .75, .95]) ]

# TeX table export
export_table(output_path, df) = open(output_path, "w") do io
    println(io, "\\begin{tabular}{ll$(repeat("r", size(df, 2)-2))}")
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
