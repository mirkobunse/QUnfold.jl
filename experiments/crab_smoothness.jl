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
bin_edges = vcat(df_acceptance[2:end-1,:e_min], df_acceptance[end-1,:e_max])

# # Bins by Max Nöthe
#
# bin_centers = 10 .^ collect(2.8:0.2:4.4)
# bin_edges = 10 .^ collect(2.7:0.2:4.5)



# measure the curvature
C = length(bin_centers)
T = LinearAlgebra.diagm( # Tikhonov matrix
    -1 => fill(-1, C-1),
    0 => fill(2, C),
    1 => fill(-1, C-1)
)[2:(C-1), :]
curvature(p) = sum([ sum(T[i, j] * p[j] for j in 1:C)^2 for i in 1:(C-2) ])



# sample acceptance-corrected probabilities from Poisson distributions
function sample_poisson(N)
    y = magic_crab_flux(bin_centers)
    y_acc = y .* df_acceptance[2:end-1,:a_eff]
    λ = y_acc * N ./ sum(y_acc) # Poisson rates for N events in total
    random_sample = [ rand(Poisson(λ_i)) for λ_i in λ ]
    p = random_sample ./ df_acceptance[2:end-1,:a_eff]
    return p ./ sum(p) # probability
end

function sample_npp(N)
    y = magic_crab_flux(bin_centers)
    y_acc = y .* df_acceptance[2:end-1,:a_eff]
    p = round_Np(N, y_acc ./ sum(y_acc)) ./ df_acceptance[2:end-1,:a_eff]
    return p ./ sum(p) # probability
end

sample_app() = rand(Dirichlet(ones(C)))

function sample_app_oq(N, m=10000, keep=.2)
    app = [ round_Np(N, sample_app()) ./ N for _ in 1:(round(Int, m/keep)) ]
    c = [ curvature(log10.(x)) for x in app ]
    i = sortperm(c)[1:m]
    return app[i]
end

# measure the curvatures of random Poisson samples and of random APP samples
poisson_curvatures(N, m=10000) = [ curvature(log10.(round_Np(N, sample_poisson(N)) ./ N)) for _ in 1:m ]
npp_curvatures(N, m=10000) = [ curvature(log10.(round_Np(N, sample_npp(N)) ./ N)) for _ in 1:m ]
app_curvatures(N, m=10000) = [ curvature(log10.(round_Np(N, sample_app()) ./ N)) for _ in 1:m ]
app_oq_curvatures(N, m=10000; keep=.2) = [ curvature(log10.(round_Np(N, x) ./ N)) for x in sample_app_oq(N, m, keep) ]

# statistics of curvatures
statistics(curvatures) = quantile(curvatures, [.05, .25, .5, .75, .95])
format_statistics(curvatures) = [ @sprintf("%.4f", x) for x ∈ statistics(curvatures) ]

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



function main(output_path="results/crab_smoothness.tex")
    df = DataFrame(
        Symbol("N") => Int[],
        Symbol("protocol") => String[],
        Symbol("5\\textsuperscript{th}") => String[],
        Symbol("25\\textsuperscript{th}") => String[],
        Symbol("50\\textsuperscript{th}") => String[],
        Symbol("75\\textsuperscript{th}") => String[],
        Symbol("95\\textsuperscript{th}") => String[],
    )
    for N in [1000, 10000]
        push!(df, vcat(N, "NPP", format_statistics(npp_curvatures(N))))
        push!(df, vcat(N, "APP", format_statistics(app_curvatures(N))))
        push!(df, vcat(N, "APP-OQ (20\\%)", format_statistics(app_oq_curvatures(N; keep=.2))))
        push!(df, vcat(N, "APP-OQ (5\\%)", format_statistics(app_oq_curvatures(N; keep=.05))))
        push!(df, vcat(N, "APP-OQ (1\\%)", format_statistics(app_oq_curvatures(N; keep=.01))))
        push!(df, vcat(N, "Poisson", format_statistics(poisson_curvatures(N))))
    end
    export_table(output_path, df)
    @info "LaTeX table exported to $(output_path)"
end
