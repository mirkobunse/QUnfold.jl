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

function plot_histogram(p, legendentry="") # plot a dis-continuous step function
    x = [NaN]
    y = [NaN]
    for i in 1:length(p)
        push!(x, QUnfoldExperiments.bin_edges()[i], QUnfoldExperiments.bin_edges()[i+1], NaN)
        push!(y, p[i], p[i], NaN)
    end
    return Plots.Linear(x, y; style="mark=none, unbounded coords=jump", legendentry=legendentry)
end

plot_error_bars(y_high, y_low) = # plot bars at QUnfoldExperiments.bin_edges() from y_high to y_low
    [ Plots.Command("\\draw[|-|] ($x,$h) -- ($x,$l)") for (x, h, l) in zip(QUnfoldExperiments.bin_centers(), y_high, y_low) ]

# function plot_poisson_sample(N) # add and subtract one Poisson std to obtain error bars
#     y = QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers())
#     p = y ./ QUnfoldExperiments.acceptance_factors()
#     λ = p * N ./ sum(p) # Poisson rates for N events in total
#     return plot_error_bars(
#         (λ + sqrt.(λ)) ./ N .* sum(p) .* QUnfoldExperiments.acceptance_factors(), # y_high
#         (λ - sqrt.(λ)) ./ N .* sum(p) .* QUnfoldExperiments.acceptance_factors() # y_low
#     )
# end

# simulated, labeled data
df = DataFrames.disallowmissing!(CSV.read("data/fact/fact_wobble.csv", DataFrame))
y = encode(
    LinearDiscretizer(log10.(QUnfoldExperiments.bin_edges())),
    df[!, :log10_energy]
)
p_trn = [ sum(y .== i) / length(y) for i in 1:length(QUnfoldExperiments.bin_centers()) ]
training_spectrum = p_trn .* sum(QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers()) ./ QUnfoldExperiments.acceptance_factors()) .* QUnfoldExperiments.acceptance_factors()

function main(output_path::String="results/crab_spectrum.pdf")
    Random.seed!(876) # make this experiment reproducible

    # read the training set, the validation pool, and the testing pool
    X_trn, y_trn, _, _, _, _ = QUnfoldExperiments.fact_data()
    n_classes = length(unique(y_trn))
    n_features = size(X_trn, 2)
    α = 1/5 # ratio of "on" vs "off" region
    t_obs = 91 # 17.7 # observation time

    X_q, X_b = QUnfoldExperiments.crab_data("data/fact/crab_mnoethe.csv")

    # TODO bootstrap the training data

    @info "Quantifying..."
    n_bins = 120
    τ_exponent = -3 # -3
    method = QUnfold.fit(HDx(
            floor(Int, n_bins / n_features);
            strategy = :softmax,
            τ = 10.0^τ_exponent,
            a = QUnfoldExperiments.acceptance_factors()
        ), X_trn, y_trn)
    p_est = QUnfold.predict_with_background(method, X_q, X_b, α)
    p_fg = QUnfold.predict(method, X_q)
    p_bg = QUnfold.predict(method, X_b)

    plot = Axis(
        Plots.Linear(QUnfoldExperiments.magic_crab_flux, (10^2.4, 10^4.8); legendentry="MAGIC (2015)"); # magic spectrum
        style = "xmode=log, ymode=log, enlarge x limits=.0425, enlarge y limits=.0425, xlabel={\$E / \\mathrm{GeV}\$}, ylabel={\$\\phi / (\\mathrm{GeV}^{-1} \\mathrm{s}^{-1} \\mathrm{m}^{-2})\$}"
    )
    plot.legendStyle = "at={(1.05,.5)}, anchor=west"
    push!(plot, plot_histogram(p_est ./ (t_obs * 60 * 60 / size(X_q, 1)) .* QUnfoldExperiments.acceptance_factors(), "est"))
    push!(plot, plot_histogram(p_fg ./ (t_obs * 60 * 60 / size(X_q, 1)) .* QUnfoldExperiments.acceptance_factors(), "fg"))
    push!(plot, plot_histogram(p_bg ./ (t_obs * 60 * 60 / size(X_q, 1)) .* QUnfoldExperiments.acceptance_factors(), "bg"))
    # push!(plot, plot_histogram(QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers())))
    # push!(plot, plot_histogram(QUnfoldExperiments.magic_crab_flux(QUnfoldExperiments.bin_centers()) ./ QUnfoldExperiments.acceptance_factors()))
    # push!(plot, plot_poisson_sample(5000)...)
    # push!(plot, plot_histogram(training_spectrum))
    save(output_path, plot)
    @info "Stored results at $(output_path)"
end
