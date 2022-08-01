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

function plot_histogram(p, legendentry) # plot a dis-continuous step function
    x = [NaN]
    y = [NaN]
    for i in 1:length(p)
        push!(x, QUnfoldExperiments.bin_edges()[i], QUnfoldExperiments.bin_edges()[i+1], NaN)
        push!(y, p[i], p[i], NaN)
    end
    return Plots.Linear(x, y; style="mark=none, unbounded coords=jump", legendentry=legendentry)
end

# simulated, labeled data
df = DataFrames.disallowmissing!(CSV.read("data/fact/fact_wobble.csv", DataFrame))
y = encode(
    LinearDiscretizer(log10.(QUnfoldExperiments.bin_edges())),
    df[!, :log10_energy]
)
p_sim = [ sum(y .== i) / length(y) for i in 1:length(QUnfoldExperiments.bin_centers()) ]
# training_spectrum = p_sim .* sum(QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers()) ./ QUnfoldExperiments.acceptance_factors()) .* QUnfoldExperiments.acceptance_factors()

# Crab parametrization with acceptance correction
p_crab = QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers()) ./ QUnfoldExperiments.acceptance_factors()
p_crab ./= sum(p_crab) # normalize to probabilities

function main()
    plot = Axis([
            plot_histogram(p_crab, "NPP (Crab)"),
            plot_histogram(p_sim, "NPP (simulation)") # plot_histogram(training_spectrum, ...)
        ],
        style = "xmode=log, ymode=log, xlabel={\$E / \\mathrm{GeV}\$}, ylabel={\$\\mathbf{P}(E)\$}"
    )
    plot.legendStyle = "at={(1.05,.5)}, anchor=west, draw=none"
    save("results/npp.pdf", plot)
    save("results/npp.tex", plot)
end
