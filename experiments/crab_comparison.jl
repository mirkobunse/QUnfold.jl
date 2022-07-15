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
DecisionTreeClassifier = pyimport_conda("sklearn.tree", "scikit-learn").DecisionTreeClassifier

function evaluate_methods(methods, X_pool, y_pool, n_samples, clf, best=Dict{Tuple{Int64,String},Vector{String}}())
    df = DataFrame(; # result storage
        N = Int[],
        protocol = String[],
        sample_index = Int[],
        method_id = String[],
        method_name = String[],
        nmd = Float64[],
        exception = String[],
    )
    for N in [1000, 10000] # the numbers of data items in each sample
        for (protocol, samples) in [
                "APP-OQ (1\\%)" => QUnfoldExperiments.sample_app_oq(N, n_samples, .01),
                "NPP (Crab)" => QUnfoldExperiments.sample_npp_crab(N, n_samples),
                "Poisson" => QUnfoldExperiments.sample_poisson(N, n_samples),
                ]
            current_methods = length(best) > 0 ? filter(x -> x[2] ∈ best[(N, protocol)], methods) : methods

            # evaluate all samples and measure the total time needed
            duration = @elapsed for (sample_index, p_true) in enumerate(samples)

                # draw a sample (X_p, y_p) from the pool, according to p_true
                i_p = QUnfoldExperiments.subsample_indices(N, p_true, y_pool)
                X_p = X_pool[i_p,:]
                QUnfoldExperiments._cache!(clf, X_p) # cache predictions

                outcomes = Array{Vector{Any}}(undef, length(current_methods))
                Threads.@threads for i_method in 1:length(current_methods)
                    id, name, method = current_methods[i_method] # un-pack the method tuple
                    outcome = [ N, protocol, sample_index, id, name ]
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
            @info "Evaluated $(n_samples) samples in $(duration) seconds" N protocol length(current_methods)
        end
    end
    return df
end

function main(;
        app_oq_path :: String = "results/crab_comparison_app_oq.tex",
        npp_crab_path :: String = "results/crab_comparison_npp_crab.tex",
        poisson_path :: String = "results/crab_comparison_poisson.tex",
        test_path :: String = "results/crab_comparison_test.csv",
        validation_path :: String = "results/crab_comparison_validation.csv",
        )
    Random.seed!(876) # make this experiment reproducible

    # read the training set, the validation pool, and the testing pool
    X_trn, y_trn, X_val, y_val, X_tst, y_tst = QUnfoldExperiments.fact_data()
    n_classes = length(unique(y_trn))
    n_features = size(X_trn, 2)

    clf = QUnfoldExperiments.CachedClassifier(
        RandomForestClassifier(24; oob_score=true, random_state=rand(UInt32), n_jobs=-1)
    )
    @info "Fitting the base classifier to $(length(y_trn)) training items" clf
    ScikitLearn.fit!(clf, X_trn, y_trn)

    methods = []
    for τ_exponent ∈ [3, 1, -1, -3] # τ = 10 ^ τ_exponent
        push!(methods, # add methods that have τ as a hyper-parameter
            ("o-acc", "o-ACC (softmax, \$\\tau=10^{$(τ_exponent)}\$)", ACC(clf; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors(), fit_classifier=false)),
            ("o-pacc", "o-PACC (softmax, \$\\tau=10^{$(τ_exponent)}\$)", PACC(clf; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors(), fit_classifier=false)),
        )
    end
    for n_bins ∈ [60, 120]
        tree_clf = QUnfoldExperiments.CachedClassifier(
            DecisionTreeClassifier(; max_leaf_nodes=n_bins, random_state=rand(UInt32));
            only_apply = true # only store tree.apply(X), do not allow predictions
        )
        t_tree = QUnfold.fit(TreeTransformer(tree_clf), X_trn, y_trn)
        push!(methods, # add methods that have n_bins as a hyper-parameter
            ("hdx", "HDx (constrained, \$B=$(n_bins)\$)", HDx(floor(Int, n_bins / n_features); strategy=:constrained)),
            ("hdy", "HDy (constrained, \$B=$(n_bins)\$)", HDy(clf, floor(Int, n_bins / n_classes); strategy=:constrained, fit_classifier=false)),
        )
        for τ_exponent ∈ [3, 1, -1, -3]
            push!(methods, # add methods that have n_bins and τ as hyper-parameters
                ("svd-original", "SVD (CONSTRAINED, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", QUnfold.SVD(t_tree; strategy=:constrained, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors())), # TODO: replace with original SVD
                ("run-softmax", "RUN (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", QUnfold.RUN(t_tree; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors())),
                ("svd-softmax", "SVD (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", QUnfold.SVD(t_tree; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors())),
                ("o-hdx", "o-HDx (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", HDx(floor(Int, n_bins / n_features); strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors())),
                ("o-hdy", "o-HDy (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", HDy(clf, floor(Int, n_bins / n_classes); strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors(), fit_classifier=false)),
            )
        end
        for n_df ∈ [10, 8, 6]
            push!(methods,
                ("run-original", "RUN (original, \$B=$(n_bins), n_{\\mathrm{df}}=$(n_df)\$)", QUnfold.RUN(t_tree; strategy=:original, n_df=n_df, a=QUnfoldExperiments.acceptance_factors()))
            )
        end
    end
    # TODO add o-SLD and IBU

    @info "Fitting $(length(methods)) methods"
    methods = [ (id, name, QUnfold.fit(method, X_trn, y_trn)) for (id, name, method) ∈ methods ]

    @info "Validating for hyper-parameter optimization"
    df = evaluate_methods(methods, X_val, y_val, 20, clf) # validate on 20 samples; TODO increase to 1000

    # store validation results
    mkpath(dirname(validation_path))
    CSV.write(validation_path, df)
    @info "$(nrow(df)) results written to $(validation_path)"

    @info "Selecting the best hyper-parameters"
    df_best = combine(
        groupby( # group average NMDs by method_id
            combine(
                groupby(df[df[!,:exception].=="",:], [:N, :protocol, :method_id, :method_name]),
                :nmd => DataFrames.mean => :nmd
            ), # average NMDs
            [:N, :protocol, :method_id]
        ),
        sdf -> begin
            sdf2 = sdf[.!(ismissing.(sdf[!,:nmd])),:]
            nrow(sdf2) > 0 ? sdf2[argmin(sdf2[!,:nmd]),:] : sdf[1,:]
        end
    )

    # fit additional methods that have no hyper-parameters
    new_methods = []
    for (id, name, method) ∈ [
            ("acc", "ACC (constrained)", ACC(clf; strategy=:constrained, fit_classifier=false)),
            ("pacc", "PACC (constrained)", PACC(clf; strategy=:constrained, fit_classifier=false)),
            # TODO add SLD
            ]
        push!(new_methods, (id, name, QUnfold.fit(method, X_trn, y_trn)))
    end
    methods = vcat(methods, new_methods)
    best = Dict(
        (N, protocol) => vcat(sdf[:,:method_name], [x[2] for x ∈ new_methods])
        for ((N, protocol), sdf) ∈ pairs(groupby(df_best, [:N, :protocol]))
    )

    @info "Final testing"
    df = evaluate_methods(methods, X_tst, y_tst, 20, clf, best) # test on 20 samples; TODO increase to 1000

    # aggregate and store testing results
    mkpath(dirname(test_path))
    CSV.write(test_path, df)
    @info "$(nrow(df)) results written to $(test_path)"

    df = combine( # average NMDs
        groupby(df[df[!,:exception].=="",:], [:N, :protocol, :method_id, :method_name]),
        :nmd => DataFrames.mean => :nmd
    )
    for (protocol, protocol_path) ∈ [
            ("APP-OQ (1\\%)", app_oq_path),
            ("NPP (Crab)", npp_crab_path),
            ("Poisson", poisson_path)
            ]
        QUnfoldExperiments.export_table(
            protocol_path,
            rename(
                sort(df[df[!,:protocol] .== protocol, [:N, :method_name, :nmd]], [:N, :nmd]),
                :method_name => :method,
                :nmd => :NMD
            )
        )
        @info "Results of $(protocol) written to $(protocol_path)"
    end
    return df
end
