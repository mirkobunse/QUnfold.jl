using CSV, DataFrames, Distributions, Statistics, PGFPlots

"""
    magic_crab_flux(x)

Compute the Crab nebula flux in `GeV⋅cm²⋅s` for a vector `x` of energy values
that are given in `GeV`. This parametrization is by Aleksíc et al. (2015).
"""
magic_crab_flux(x) = @. 3.23e-10 * (x/1e3)^(-2.47 - 0.24 * log10(x/1e3))



# our bins
df_acceptance = CSV.read("data/fact/acceptance.csv", DataFrame)
bin_centers = df_acceptance[2:end-1,:bin_center]
bin_edges = vcat(df_acceptance[2:end-1,:e_min], df_acceptance[end-1,:e_max])

# # Bins by Max Nöthe
#
# bin_centers = 10 .^ collect(2.8:0.2:4.4)
# bin_edges = 10 .^ collect(2.7:0.2:4.5)

function plot_histogram(p) # plot a dis-continuous step function
    x = [NaN]
    y = [NaN]
    for i in 1:length(p)
        push!(x, bin_edges[i], bin_edges[i+1], NaN)
        push!(y, p[i], p[i], NaN)
    end
    return Plots.Linear(x, y; style="mark=none, unbounded coords=jump")
end

function sample_poisson(N) # add and subtract one Poisson std to obtain error bars
    y = magic_crab_flux(bin_centers)
    y_acc = y .* df_acceptance[2:end-1,:a_eff]
    λ = y_acc * (N-length(y)) ./ sum(y_acc) # Poisson rates for N events in total
    random_sample = [ 1 + rand(Poisson(λ_i)) for λ_i in λ ]
    p = random_sample ./ df_acceptance[2:end-1,:a_eff] ./ sum(random_sample ./ df_acceptance[2:end-1,:a_eff]) # probability
    @info "Probability" p log10.(p)
    return Plots.Linear(1:length(p), log10.(p))
end



function main()
    plot = Axis([sample_poisson(1000) for _ in 1:3])
    save("results/crab_smoothness.pdf", plot)
end
