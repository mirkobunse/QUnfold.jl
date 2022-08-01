if ".." ∉ LOAD_PATH push!(LOAD_PATH, "..") end # add QUnfold to the LOAD_PATH
ENV["PYTHONWARNINGS"] = "ignore"
using
    ArgParse,
    CSV,
    DataFrames,
    DelimitedFiles,
    Discretizers,
    Distributions,
    LinearAlgebra,
    Printf,
    PyCall,
    PGFPlots,
    QUnfold,
    QUnfoldExperiments,
    Random,
    Statistics,
    StatsBase,
    QuaPy
import ScikitLearn, ScikitLearnBase

RandomForestClassifier = pyimport_conda("sklearn.ensemble", "scikit-learn").RandomForestClassifier
DecisionTreeClassifier = pyimport_conda("sklearn.tree", "scikit-learn").DecisionTreeClassifier

function plot_histogram(p) # plot a dis-continuous step function
    x = [NaN]
    y = [NaN]
    for i in 1:length(p)
        push!(x, QUnfoldExperiments.bin_edges()[i], QUnfoldExperiments.bin_edges()[i+1], NaN)
        push!(y, p[i], p[i], NaN)
    end
    return Plots.Linear(x, y; style="mark=none, unbounded coords=jump")
end

plot_error_bars(y_high, y_low) = # plot bars at QUnfoldExperiments.bin_edges() from y_high to y_low
    [ Plots.Command("\\draw[|-|] ($x,$h) -- ($x,$l)") for (x, h, l) in zip(QUnfoldExperiments.bin_centers(), y_high, y_low) ]

function plot_poisson_sample(N) # add and subtract one Poisson std to obtain error bars
    y = QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers())
    p = y ./ QUnfoldExperiments.acceptance_factors()
    λ = p * N ./ sum(p) # Poisson rates for N events in total
    return plot_error_bars(
        (λ + sqrt.(λ)) ./ N .* sum(p) .* QUnfoldExperiments.acceptance_factors(), # y_high
        (λ - sqrt.(λ)) ./ N .* sum(p) .* QUnfoldExperiments.acceptance_factors() # y_low
    )
end

# simulated, labeled data
df = DataFrames.disallowmissing!(CSV.read("data/fact/fact_wobble.csv", DataFrame))
y = encode(
    LinearDiscretizer(log10.(QUnfoldExperiments.bin_edges())),
    df[!, :log10_energy]
)
p_trn = [ sum(y .== i) / length(y) for i in 1:length(QUnfoldExperiments.bin_centers()) ]
training_spectrum = p_trn .* sum(QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers()) ./ QUnfoldExperiments.acceptance_factors()) .* QUnfoldExperiments.acceptance_factors()

function main()
    plot = Axis(
        Plots.Linear(QUnfoldExperiments.magic_crab_flux, (10^2.4, 10^4.8)); # magic spectrum
        style = "xmode=log, ymode=log, enlarge x limits=.0425, enlarge y limits=.0425"
    )
    push!(plot, plot_histogram(QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers())))
    # push!(plot, plot_histogram(QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers()) ./ QUnfoldExperiments.acceptance_factors()))
    push!(plot, plot_poisson_sample(5000)...)
    push!(plot, plot_histogram(training_spectrum))
    save("results/crab_spectrum.pdf", plot)
end
