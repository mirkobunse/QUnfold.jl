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

plot_error_bars(y_high, y_low) = # plot bars at bin_edges from y_high to y_low
    [ Plots.Command("\\draw[|-|] ($x,$h) -- ($x,$l)") for (x, h, l) in zip(bin_centers, y_high, y_low) ]

function plot_poisson_sample(N) # add and subtract one Poisson std to obtain error bars
    y = magic_crab_flux(bin_centers)
    p = y .* df_acceptance[2:end-1,:a_eff]
    λ = p * N ./ sum(p) # Poisson rates for N events in total
    return plot_error_bars(
        (λ + sqrt.(λ)) ./ N .* sum(p) ./ df_acceptance[2:end-1,:a_eff], # y_high
        (λ - sqrt.(λ)) ./ N .* sum(p) ./ df_acceptance[2:end-1,:a_eff] # y_low
    )
end



function main()
    plot = Axis(
        Plots.Linear(magic_crab_flux, (10^2.4, 10^4.8)); # magic spectrum
        style = "xmode=log, ymode=log, enlarge x limits=.0425, enlarge y limits=.0425"
    )
    push!(plot, plot_histogram(magic_crab_flux(bin_centers)))
    # push!(plot, plot_histogram(magic_crab_flux(bin_centers) .* df_acceptance[2:end-1,:a_eff]))
    push!(plot, plot_poisson_sample(5000)...)
    save("results/crab_spectrum.pdf", plot)
end
