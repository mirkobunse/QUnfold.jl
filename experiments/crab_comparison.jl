if ".." ∉ LOAD_PATH push!(LOAD_PATH, "..") end # add QUnfold to the LOAD_PATH
ENV["PYTHONWARNINGS"] = "ignore"
using
    ArgParse,
    CSV,
    DataFrames,
    DelimitedFiles,
    LinearAlgebra,
    Printf,
    PyCall,
    QUnfold,
    QUnfoldExperiments,
    Random,
    StatsBase,
    QuaPy
import ScikitLearn, ScikitLearnBase

RandomForestClassifier = pyimport_conda("sklearn.ensemble", "scikit-learn").RandomForestClassifier
DecisionTreeClassifier = pyimport_conda("sklearn.tree", "scikit-learn").DecisionTreeClassifier

function evaluate_methods(N, methods, X_pool, y_pool, n_samples, classifiers, best=Dict{String,Vector{String}}())
    df = DataFrame(; # result storage
        protocol = String[],
        sample_index = Int[],
        method_id = String[],
        method_name = String[],
        nmd = Float64[],
        exception = String[],
    )
    for (protocol, samples) in [
            "APP-OQ (1\\%)" => QUnfoldExperiments.sample_app_oq(N, n_samples, .01),
            "NPP (Crab)" => QUnfoldExperiments.sample_npp_crab(N, n_samples),
            "Poisson" => QUnfoldExperiments.sample_poisson(N, n_samples),
            ]
        current_methods = length(best) > 0 ? filter(x -> x[2] ∈ best[protocol], methods) : methods

        # evaluate all samples and measure the total time needed
        duration = @elapsed for (sample_index, p_true) in enumerate(samples)

            # draw a sample (X_p, y_p) from the pool, according to p_true
            i_p = QUnfoldExperiments.subsample_indices(N, p_true, y_pool)
            X_p = X_pool[i_p,:]
            for clf ∈ classifiers
                QUnfoldExperiments._cache!(clf, X_p) # cache predictions
            end

            outcomes = Array{Vector{Any}}(undef, length(current_methods))
            Threads.@threads for i_method in 1:length(current_methods)
                id, name, method = current_methods[i_method] # un-pack the method tuple
                outcome = [ protocol, sample_index, id, name ]
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
        @info "Evaluated $(length(current_methods)) methods on $(n_samples) samples in $(duration) seconds" N protocol
    end
    return df
end

function main(;
        table_path :: String = "results/crab_comparison_01k.tex",
        N :: Int = 1000,
        test_path :: String = "results/crab_comparison_01k_test.csv",
        validation_path :: String = "results/crab_comparison_01k_validation.csv",
        read_validation :: Bool = false,
        read_test :: Bool = false,
        is_test_run :: Bool = false
        )
    Random.seed!(876) # make this experiment reproducible

    df = DataFrame()
    if read_test
        @info "Reading test results from $(test_path)"
        df = CSV.read(test_path, DataFrame)
        df[!,:exception] = coalesce.(df[!,:exception], "")
    else
        # read the training set, the validation pool, and the testing pool
        X_trn, y_trn, X_val, y_val, X_tst, y_tst = QUnfoldExperiments.fact_data()
        @info "Read $(length(y_val)) validation and $(length(y_tst)) test items"
        n_classes = length(unique(y_trn))
        n_features = size(X_trn, 2)

        clf = QUnfoldExperiments.CachedClassifier(
            RandomForestClassifier(is_test_run ? 24 : 100; criterion="entropy", oob_score=true, random_state=rand(UInt32), n_jobs=-1)
        )
        @info "Fitting the base classifier to $(length(y_trn)) training items" clf
        ScikitLearn.fit!(clf, X_trn, y_trn)

        methods = []
        classifiers = [ clf ]
        for τ_exponent ∈ (is_test_run ? [-1] : [3, 1, -1, -3]) # τ = 10 ^ τ_exponent
            push!(methods, # add methods that have τ as a hyper-parameter
                ("o-acc", "o-ACC (softmax, \$\\tau=10^{$(τ_exponent)}\$)", ACC(clf; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors(), fit_classifier=false)),
                ("o-pacc", "o-PACC (softmax, \$\\tau=10^{$(τ_exponent)}\$)", PACC(clf; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors(), fit_classifier=false)),
            )
        end
        for n_bins ∈ (is_test_run ? [60] : [60, 120])
            tree_clf = QUnfoldExperiments.CachedClassifier(
                DecisionTreeClassifier(; criterion="entropy", max_leaf_nodes=n_bins, random_state=rand(UInt32));
                only_apply = true # only store tree.apply(X), do not allow predictions
            )
            t_tree = QUnfold.fit(TreeTransformer(tree_clf), X_trn, y_trn)
            push!(classifiers, t_tree.tree)
            push!(methods, # add methods that have n_bins as a hyper-parameter
                ("hdx", "HDx (constrained, \$B=$(n_bins)\$)", HDx(floor(Int, n_bins / n_features); strategy=:constrained)),
                ("hdy", "HDy (constrained, \$B=$(n_bins)\$)", HDy(clf, floor(Int, n_bins / n_classes); strategy=:constrained, fit_classifier=false)),
            )
            for τ_exponent ∈ (is_test_run ? [-1] : [3, 1, -1, -3])
                push!(methods, # add methods that have n_bins and τ as hyper-parameters
                    ("run-softmax", "RUN (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", QUnfold.RUN(t_tree; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors())),
                    ("svd-softmax", "SVD (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", QUnfold.SVD(t_tree; strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors())),
                    ("o-hdx", "o-HDx (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", HDx(floor(Int, n_bins / n_features); strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors())),
                    ("o-hdy", "o-HDy (softmax, \$B=$(n_bins), \\tau=10^{$(τ_exponent)}\$)", HDy(clf, floor(Int, n_bins / n_classes); strategy=:softmax, τ=10.0^τ_exponent, a=QUnfoldExperiments.acceptance_factors(), fit_classifier=false)),
                )
            end
            for n_df ∈ (is_test_run ? [8] : [10, 8, 6])
                push!(methods, # add methods that have n_bins and n_df as hyper-parameters
                    ("run-original", "RUN (original, \$B=$(n_bins), n_{\\mathrm{df}}=$(n_df)\$)", QUnfold.RUN(t_tree; strategy=:original, n_df=n_df, a=QUnfoldExperiments.acceptance_factors())),
                    ("svd-original", "SVD (original, \$B=$(n_bins), n_{\\mathrm{df}}=$(n_df)\$)", QUnfold.SVD(t_tree; strategy=:original, n_df=n_df, a=QUnfoldExperiments.acceptance_factors())),
                )
            end
            for o ∈ (is_test_run ? [0] : [0, 1, 2]), λ ∈ (is_test_run ? [.2] : [.2, .5])
                push!(methods,
                    ("ibu", "IBU (\$B=$(n_bins), o=$(o), \\lambda=$(λ)\$)", IBU(t_tree; o=o, λ=λ, a=QUnfoldExperiments.acceptance_factors())),
                )
            end
        end
        for o ∈ (is_test_run ? [0] : [0, 1, 2]), λ ∈ (is_test_run ? [.2] : [.2, .5])
            push!(methods,
                ("o-sld", "o-SLD (\$o=$(o), \\lambda=$(λ)\$)", SLD(clf; o=o, λ=λ, a=QUnfoldExperiments.acceptance_factors(), fit_classifier=false)),
            )
        end

        @info "Fitting $(length(methods)) methods"
        methods = [ (id, name, QUnfold.fit(method, X_trn, y_trn)) for (id, name, method) ∈ methods ]
        for clf ∈ classifiers
            setfield!(clf, :cold_cache_warnings, true)
        end

        df = DataFrame()
        if read_validation
            @info "Reading validation results from $(validation_path)"
            df = CSV.read(validation_path, DataFrame)
            df[!,:exception] = coalesce.(df[!,:exception], "")
        else
            @info "Validating for hyper-parameter optimization"
            df = evaluate_methods(N, methods, X_val, y_val, is_test_run ? 5 : 1000, classifiers)

            # store validation results
            mkpath(dirname(validation_path))
            CSV.write(validation_path, df)
            @info "$(nrow(df)) results written to $(validation_path)"
        end

        @info "Selecting the best hyper-parameters"
        df_best = combine(
            groupby( # group average NMDs by method_id
                combine(
                    groupby(df[df[!,:exception].=="",:], [:protocol, :method_id, :method_name]),
                    :nmd => DataFrames.mean => :nmd
                ), # average NMDs
                [:protocol, :method_id]
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
                ("sld", "SLD", SLD(clf, fit_classifier=false)),
                ]
            push!(new_methods, (id, name, QUnfold.fit(method, X_trn, y_trn)))
        end
        methods = vcat(methods, new_methods)
        best = Dict(
            protocol => vcat(sdf[:,:method_name], [x[2] for x ∈ new_methods])
            for ((protocol,), sdf) ∈ pairs(groupby(df_best, :protocol))
        )

        @info "Final testing"
        df = evaluate_methods(N, methods, X_tst, y_tst, is_test_run ? 5 : 1000, classifiers, best)

        # aggregate and store testing results
        mkpath(dirname(test_path))
        CSV.write(test_path, df)
        @info "$(nrow(df)) results written to $(test_path)"
    end # if read_test

    # aggregation and export
    df = combine( # average NMDs
        groupby(df[df[!,:exception].=="",:], [:protocol, :method_id, :method_name]),
        :nmd => DataFrames.mean => :nmd,
        :nmd => DataFrames.std => :std
    )
    df[!,:value] = [
        "\$" * @sprintf("%.4f", x)[2:end] * "\\pm" * @sprintf("%.4f", y)[2:end] * "\$"
        for (x, y) ∈ zip(df[!,:nmd], df[!,:std])
    ]
    df[!,:method] = [
        get(Dict( # leading number allows ordering; will be removed
                "run-original" => "01 RUN (original)",
                "svd-original" => "02 SVD (original)",
                "ibu" => "03 IBU",
                "acc" => "04 ACC (constrained)",
                "pacc" => "05 PACC (constrained)",
                "hdx" => "06 HDx (constrained)",
                "hdy" => "07 HDy (constrained)",
                "run-softmax" => "08 RUN (softmax)",
                "svd-softmax" => "09 SVD (softmax)",
                "o-acc" => "10 o-ACC (softmax)",
                "o-pacc" => "11 o-PACC (softmax)",
                "o-hdx" => "12 o-HDx (softmax)",
                "o-hdy" => "13 o-HDy (softmax)",
                "o-sld" => "14 o-SLD",
                "sld" => "15 SLD"
            ), x, y)
        for (x, y) ∈ zip(df[!,:method_id], df[!,:method_name])
    ]
    df = sort( # long to wide format, sorted by method name
        unstack(df[:, [:method, :protocol, :value]], :method, :protocol, :value),
        :method
    )
    df[!,:method] = [ x[4:end] for x ∈ df[!,:method] ] # remove leading number
    for c ∈ propertynames(df)[2:end] # bold font for winning methods
        matches = match.(r"\$(.\d+)\\pm(.\d+)\$", df[!,c])
        nmd = [ parse(Float64, "0" * m[1]) for m ∈ matches ]
        std_dev = [ parse(Float64, "0" * m[2]) for m ∈ matches ]
        i_min = findall(nmd .== minimum(nmd)) # indices of all minimum average NMDs
        i = findall(nmd .> minimum(nmd) + maximum(std_dev[i_min])) # indices of losing methods
        df[i_min, c] = "\$\\mathbf{" .* [ x[2:end-1] for x ∈ df[i_min, c] ] .* "}\$"
        df[i, c] = "\\textcolor{gray}{" .* df[i, c] .* "}"
    end
    QUnfoldExperiments.export_table(table_path, df, "lccc")
    @info "LaTeX table written to $(table_path)"
    return df
end

# command line interface
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--read_validation"
            help = "whether to read the validation results instead of validating"
            action = :store_true
        "--read_test"
            help = "whether to read the test results instead of testing"
            action = :store_true
        "--is_test_run", "-t"
            help = "whether this run is shortened for testing purposes"
            action = :store_true
        "--N"
            help = "the number of items per sample, e.g., N=1000 or N=10000"
            arg_type = Int
            required = true
        "--test_path"
            help = "the output path of the test results"
            required = true
        "--validation_path"
            help = "the output path of the validation results"
            required = true
        "table_path"
            help = "the output path of the LaTeX table"
            required = true
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; parse_commandline()...)
end
