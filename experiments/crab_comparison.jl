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

function evaluate_methods(methods, X_pool, y_pool, n_samples, clf)
    df = DataFrame(; # result storage
        N = Int[],
        protocol = String[],
        sample_index = Int[],
        method = String[],
        nmd = Float64[],
        exception = String[],
    )
    for N in [1000, 10000] # the numbers of data items in each sample
        for (protocol, samples) in [
                "APP-OQ (1\\%)" => QUnfoldExperiments.sample_app_oq(N, n_samples, .01),
                "NPP (Crab)" => QUnfoldExperiments.sample_npp_crab(N, n_samples),
                "Poisson" => QUnfoldExperiments.sample_poisson(N, n_samples),
                ]
            # evaluate all samples and measure the total time needed
            duration = @elapsed for (sample_index, p_true) in enumerate(samples)

                # draw a sample (X_p, y_p) from the pool, according to p_true
                i_p = QUnfoldExperiments.subsample_indices(N, p_true, y_pool)
                X_p = X_pool[i_p,:]
                QUnfoldExperiments._cache!(clf, X_p) # cache predictions

                outcomes = Array{Vector{Any}}(undef, length(methods))
                Threads.@threads for i_method in 1:length(methods)
                    method_name, method = methods[i_method]
                    outcome = [ N, protocol, sample_index, method_name ]
                    try
                        p_hat = QUnfold.predict(method, X_p)
                        nmd = QUnfoldExperiments.nmd(
                            QUnfoldExperiments.to_log10_spectrum_density(N, p_hat),
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
                    outcomes[i_method] = outcome
                end
                push!(df, outcomes...)
            end
            @info "Evaluated $(n_samples) samples in $(duration) seconds" N protocol length(methods)
        end
    end
    return df
end

function main(;
        output_path :: String = "results/crab_comparison.csv",
        validation_path :: String = "results/crab_comparison_validation.csv",
        )
    Random.seed!(876) # make this experiment reproducible

    # read the training set, the validation pool, and the testing pool
    X_trn, y_trn, X_val, y_val, X_tst, y_tst = QUnfoldExperiments.fact_data()

    clf = QUnfoldExperiments.CachedClassifier(
        RandomForestClassifier(24; oob_score=true, random_state=rand(UInt32), n_jobs=-1)
    )
    @info "Fitting the base classifier to $(length(y_trn)) training items" clf
    ScikitLearn.fit!(clf, X_trn, y_trn)
    t_clf = ClassTransformer(clf; fit_classifier=false)

    methods = []
    for τ_exponent ∈ [3, 1, -1, -3] # τ = 10 ^ τ_exponent
        push!(methods, # add methods that have τ as a hyper-parameter
            "o-ACC (softmax, \$\\tau=10^{$(τ_exponent)}\$)" => ACC(clf; strategy=:softmax, τ=10.0^τ_exponent, fit_classifier=false),
            "o-PACC (softmax, \$\\tau=10^{$(τ_exponent)}\$)" => PACC(clf; strategy=:softmax, τ=10.0^τ_exponent, fit_classifier=false),
            "RUN (CONSTRAINED, \$\\tau=10^{$(τ_exponent)}\$)" => QUnfold.RUN(t_clf; strategy=:constrained, τ=10.0^τ_exponent), # TODO: replace with original RUN
            "SVD (CONSTRAINED, \$\\tau=10^{$(τ_exponent)}\$)" => QUnfold.SVD(t_clf; strategy=:constrained, τ=10.0^τ_exponent), # TODO: replace with original SVD
            "RUN (softmax, \$\\tau=10^{$(τ_exponent)}\$)" => QUnfold.RUN(t_clf; strategy=:softmax, τ=10.0^τ_exponent),
            "SVD (softmax, \$\\tau=10^{$(τ_exponent)}\$)" => QUnfold.SVD(t_clf; strategy=:softmax, τ=10.0^τ_exponent),
        )
        for n_bins ∈ [2, 4]
            push!(methods, # add methods that have τ and n_bins as hyper-parameters
                "o-HDx (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)" => HDx(n_bins; strategy=:softmax, τ=10.0^τ_exponent),
                "o-HDy (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)" => HDy(clf, n_bins; strategy=:softmax, τ=10.0^τ_exponent, fit_classifier=false),
            )
        end
    end
    for n_bins ∈ [2, 4]
        push!(methods, # add methods that have n_bins as a hyper-parameter
            "HDx (constrained, \$B=$(n_bins)\$)" => HDx(n_bins; strategy=:constrained),
            "HDy (constrained, \$B=$(n_bins)\$)" => HDy(clf, n_bins; strategy=:constrained, fit_classifier=false),
        )
    end
    # TODO add o-SLD and IBU
    
    @info "Fitting $(length(methods)) methods"
    methods = [ method_name => QUnfold.fit(method, X_trn, y_trn) for (method_name, method) ∈ methods ]

    @info "Validating for hyper-parameter optimization"
    df = evaluate_methods(methods, X_val, y_val, 20, clf) # validate on 20 samples; TODO increase to 1000

    # # TODO: aggregation + hyper-parameter selection
    # @info "Selecting the best hyper-parameters"
    # methods = filter(x -> is_best(x), methods)
    # 
    # # fit additional methods that have no hyper-parameters
    # for (method_name, method) ∈ [
    #         "ACC (constrained)" => ACC(clf; strategy=:constrained, fit_classifier=false),
    #         "PACC (constrained)" => PACC(clf; strategy=:constrained, fit_classifier=false),
    #         # TODO add SLD
    #         ]
    #     push!(methods, method_name => QUnfold.fit(method, X_trn, y_trn))
    # end
    # 
    # @info "Final testing"
    # df = evaluate_methods(methods, X_tst, y_tst, 20, clf) # validate on 20 samples; TODO increase to 1000
    # 
    # # TODO: aggregate and return

    return df
end
