if ".." ∉ LOAD_PATH push!(LOAD_PATH, "..") end # add QUnfold to the LOAD_PATH
ENV["PYTHONWARNINGS"] = "ignore"
using
    ArgParse,
    CSV,
    DataFrames,
    DelimitedFiles,
    LinearAlgebra,
    PyCall,
    QUnfold,
    QUnfoldExperiments,
    Random,
    StatsBase,
    QuaPy
import ScikitLearn, ScikitLearnBase

RandomForestClassifier = pyimport_conda("sklearn.ensemble", "scikit-learn").RandomForestClassifier

function evaluate_methods(methods)
    df = DataFrame(; # result storage
        N = Int[],
        protocol = String[],
        sample_index = Int[],
        method = String[],
        nmd = Float64[],
        exception = String[],
    )
    m_val = 100 # the number of validation samples; TODO increase to 1000
    m_tst = 100 # the number of test samples; TODO increase to 1000
    for N in [1000, 10000] # the numbers of data items in each sample
        for (protocol, samples) in [
                "APP-OQ (1\\%)" => QUnfoldExperiments.sample_app_oq(N, m_val, .01),
                "NPP (Crab)" => QUnfoldExperiments.sample_npp_crab(N, m_val),
                "Poisson" => QUnfoldExperiments.sample_poisson(N, m_val),
                ]
            for (sample_index, p_true) in enumerate(samples)

                # TODO: sample (X_p, y_p) from p

                for (method_name, method) ∈ methods
                    outcome = [N, protocol, sample_index, method_name, C]
                    try
                        p_hat = QUnfold.predict(method, X_p)
                        nmd = QUnfoldExperiments.nmd(
                            QUnfoldExperiments.to_log10_spectrum_density(N, p_hat)
                            QUnfoldExperiments.to_log10_spectrum_density(N, p_true)
                        )
                        push!(outcome, nmd, "")
                    catch err
                        if isa(err, QUnfold.NonOptimalStatusError)
                            push!(outcome, NaN, string(err.termination_status))
                        elseif isa(err, SingularException)
                            push!(outcome, NaN, "SingularException")
                        else
                            rethrow()
                        end
                    end
                    push!(df, outcome)
                end
            end
        end
    end
    return df
end

function main(;
        output_path :: String = "results/crab_comparison.csv",
        validation_path :: String = "results/crab_comparison_validation.csv",
        )
    Random.seed!(876) # make this experiment reproducible

    # TODO read (X_trn, y_trn) and the pools (X_val, y_val), (X_tst, y_tst)

    clf = RandomForestClassifier(; oob_score=true, random_state=rand(UInt32), n_jobs=-1)
    fit!(clf, X_trn, y_trn)

    methods = []
    for (method_name, method) ∈ [ # fit all methods
            "ACC (constrained)" => ACC(clf; strategy=:constrained, fit_classifier=false),
            "PACC (constrained)" => PACC(clf; strategy=:constrained, fit_classifier=false),
            "o-ACC (softmax, $\\tau=10^{-2}$)" => ACC(clf; strategy=:softmax, τ=1e-2, fit_classifier=false),
            "o-PACC (softmax, $\\tau=10^{-2}$)" => PACC(clf; strategy=:softmax, τ=1e-2, fit_classifier=false),
        ]
        push!(methods, method_name => QUnfold.fit(method, X_trn, y_trn))
    end

    # TODO: aggregation + hyper-parameter selection & adding of non-parametrized methods + testing
    return evaluate_methods(methods)
end
