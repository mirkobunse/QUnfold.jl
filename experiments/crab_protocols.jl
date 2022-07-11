using
    CSV,
    DataFrames,
    Distributions,
    LinearAlgebra,
    Random,
    Statistics,
    StatsBase,
    Printf

"""
    magic_crab_flux(x)

Compute the Crab nebula flux in `GeV⋅cm²⋅s` for a vector `x` of energy values
that are given in `GeV`. This parametrization is by Aleksíc et al. (2015).
"""
magic_crab_flux(x) = @. 3.23e-10 * (x/1e3)^(-2.47 - 0.24 * log10(x/1e3))

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



# our bins
df_acceptance = CSV.read("data/fact/acceptance.csv", DataFrame)
bin_centers = df_acceptance[2:end-1,:bin_center]
bin_edges = disallowmissing(vcat(df_acceptance[2:end-1,:e_min], df_acceptance[end-1,:e_max]))

# # Bins by Max Nöthe
#
# bin_centers = 10 .^ collect(2.8:0.2:4.4)
# bin_edges = 10 .^ collect(2.7:0.2:4.5)



# simulated, labeled data
df = DataFrames.disallowmissing!(CSV.read("data/fact/fact_wobble.csv", DataFrame))
y = encode(
    LinearDiscretizer(log10.(bin_edges)),
    df[!, :log10_energy]
)
p_trn = [ sum(y .== i) / length(y) for i in 1:length(bin_centers) ]



# measure the curvature 1/2 * (Tp)^2
C = length(bin_centers)
T = LinearAlgebra.diagm( # Tikhonov matrix
    -1 => fill(-1, C-1),
    0 => fill(2, C),
    1 => fill(-1, C-1)
)[2:(C-1), :]
curvature(p) = sum([ sum(T[i, j] * p[j] for j in 1:C)^2 for i in 1:(C-2) ]) / 2



# sample acceptance-corrected probabilities from Poisson distributions
sample_poisson(N, m) = [ sample_poisson(N) for _ in 1:m ]
function sample_poisson(N)
    y = magic_crab_flux(bin_centers)
    y_acc = y .* df_acceptance[2:end-1,:a_eff]
    λ = y_acc * N ./ sum(y_acc) # Poisson rates for N events in total
    random_sample = [ rand(Poisson(λ_i)) for λ_i in λ ]
    p = round_Np(N, random_sample ./ sum(random_sample)) ./ df_acceptance[2:end-1,:a_eff]
    return p ./ sum(p) # probability
end

sample_npp_crab(N, m) = [ sample_npp_crab(N) for _ in 1:m ]
function sample_npp_crab(N)
    y = magic_crab_flux(bin_centers)
    y_acc = y .* df_acceptance[2:end-1,:a_eff]
    p = round_Np(N, y_acc ./ sum(y_acc)) ./ df_acceptance[2:end-1,:a_eff]
    return p ./ sum(p) # probability
end

sample_npp_simulation(N, m) = [ sample_npp_simulation(N) for _ in 1:m ]
sample_npp_simulation(N) = round_Np(N, p_trn) ./ N

sample_app(N, m) = [ sample_app(N) for _ in 1:m ]
sample_app(N) = round_Np(N, rand(Dirichlet(ones(C)))) ./ N

function sample_app_oq(N, m=10000, keep=.2)
    app = [ round_Np(N, sample_app()) ./ N for _ in 1:(round(Int, m/keep)) ]
    c = [ curvature(log10.(x)) for x in app ]
    i = sortperm(c)[1:m]
    return app[i]
end



spectrum_nmd(N::Int, a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    nmd(to_log10_spectrum_density(N, a), to_log10_spectrum_density(N, b))

# map class prevalences to spectra
function to_log10_spectrum_density(N::Int, a::AbstractVector{T}) where T<:Number
    p = a ./ df_acceptance[2:end-1,:a_eff]
    p = log10.(1 .+ p ./ sum(p) .* (N-length(p)))
    return p ./ sum(p)
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
format_statistics(x) = [ @sprintf("%.4f", x) for x ∈ quantile(x, [.05, .25, .5, .75, .95]) ]

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



function main(;
        curvature_path="results/crab_protocols_Tp.tex",
        shift_path="results/crab_protocols_nmd.tex"
    )
    df_curvature = DataFrame(
        Symbol("N") => Int[],
        Symbol("protocol") => String[],
        Symbol("5\\textsuperscript{th}") => String[],
        Symbol("25\\textsuperscript{th}") => String[],
        Symbol("50\\textsuperscript{th}") => String[],
        Symbol("75\\textsuperscript{th}") => String[],
        Symbol("95\\textsuperscript{th}") => String[],
    )
    df_shift = copy(df_curvature) # same columns
    m = 10000 # number of samples generated by each protocol
    for N in [1000, 10000]
        for (protocol, samples) in [
                "APP" => sample_app(N, m),
                "APP-OQ (20\\%)" => sample_app_oq(N, m, .2),
                "APP-OQ (5\\%)" => sample_app_oq(N, m, .05),
                "APP-OQ (1\\%)" => sample_app_oq(N, m, .01),
                "NPP (Crab)" => sample_npp_crab(N, m),
                "NPP (simulation)" => sample_npp_simulation(N, m),
                "Poisson" => sample_poisson(N, m),
            ]
            push!(df_curvature, vcat(
                N,
                protocol,
                format_statistics([ try curvature(log10.(p)); catch; @error "curvature" p protocol end for p in samples])
            ))
            push!(df_shift, vcat(
                N,
                protocol,
                format_statistics([ spectrum_nmd(N, p, p_trn) for p in samples])
            ))
        end
    end
    export_table(curvature_path, df_curvature)
    export_table(shift_path, df_shift)
    @info "LaTeX tables exported to $(curvature_path) and $(shift_path)"
end
