if ".." ∉ LOAD_PATH push!(LOAD_PATH, "..") end # add QUnfold to the LOAD_PATH
ENV["PYTHONWARNINGS"] = "ignore"
using
    ArgParse,
    Conda,
    CSV,
    DataFrames,
    DelimitedFiles,
    LinearAlgebra,
    PyCall,
    QUnfold,
    Random,
    StatsBase,
    QuaPy
import ScikitLearn, ScikitLearnBase, QuaPy

Random.seed!(42) # make tests reproducible


# classifiers and data sets

BaggingClassifier = pyimport_conda("sklearn.ensemble", "scikit-learn").BaggingClassifier
LogisticRegression = pyimport_conda("sklearn.linear_model", "scikit-learn").LogisticRegression

function read_trn(path="data/T1B/public/training_data.txt")
    data = readdlm(path, ',', '\n'; skipstart=1)
    return data[:,2:end], round.(Int, data[:,1]) .+ 1 # = X, y
end
read_prevalences(path) = readdlm(path, ',', '\n'; skipstart=1)[:, 2:end]
read_sample(index, dir) = readdlm("$(dir)/$(index).txt", ',', '\n'; skipstart=1)


# utilities for QuaPy wrappers

abstract type _QuaPyMethod <: QUnfold.AbstractMethod end

struct _FittedQuaPyMethod
    quantifier::QuaPy.Methods.BaseQuantifier
end

QUnfold.predict(m::_FittedQuaPyMethod, X::Any) = QuaPy.quantify(m.quantifier, X)


# QuaPy EMQ

struct _QuaPyEMQ <: _QuaPyMethod
    classifier::Any
    fit_classifier::Bool
end
QuaPyEMQ(c::Any; fit_classifier::Bool=true) = _QuaPyEMQ(c, fit_classifier)

function QUnfold.fit(m::_QuaPyEMQ, X::Any, y::AbstractVector{T}) where {T <: Integer}
    quantifier = QuaPy.Methods.EMQ(m.classifier)
    QuaPy.fit!(quantifier, X, y; fit_learner=m.fit_classifier)
    return _FittedQuaPyMethod(quantifier)
end


# QuaPy ACC / PACC

struct _QuaPyACC <: _QuaPyMethod
    classifier::Any
    is_probabilistic::Bool
    oob_score::Bool
    fit_classifier::Bool
end
QuaPyACC(c::Any; oob_score::Bool=true, fit_classifier::Bool=true) =
    _QuaPyACC(c, false, oob_score, fit_classifier)
QuaPyPACC(c::Any; oob_score::Bool=true, fit_classifier::Bool=true) =
    _QuaPyACC(c, true, oob_score, fit_classifier)

py"""
import numpy as np
from sklearn.metrics import confusion_matrix

# https://github.com/HLT-ISTI/QuaPy/blob/6a5c528154c2d6d38d9f3258e667727bf692fc8b/quapy/method/aggregative.py#L319
def accPteCondEstim(classes, y, y_):
    conf = confusion_matrix(y, y_, labels=classes).T
    conf = conf.astype(np.float)
    class_counts = conf.sum(axis=0)
    for i, _ in enumerate(classes):
        if class_counts[i] == 0:
            conf[i, i] = 1
        else:
            conf[:, i] /= class_counts[i]
    return conf

# https://github.com/HLT-ISTI/QuaPy/blob/6a5c528154c2d6d38d9f3258e667727bf692fc8b/quapy/method/aggregative.py#L450
def paccPteCondEstim(classes, y, y_):
    confusion = np.eye(len(classes))
    for i, class_ in enumerate(classes):
        idx = y == class_
        if idx.any():
            confusion[i] = y_[idx].mean(axis=0)
    return confusion.T
"""

function QUnfold.fit(m::_QuaPyACC, X::Any, y::AbstractVector{T}) where {T <: Integer}
    if m.oob_score # custom ACC/PACC.fit with OOB decision function
        classifier = m.classifier
        if !hasproperty(classifier, :oob_score) || !classifier.oob_score
            error("Only bagging classifiers with oob_score=true are supported")
        end # TODO add support for non-bagging classifiers
        if m.fit_classifier
            classifier = ScikitLearnBase.clone(classifier)
            ScikitLearnBase.fit!(classifier, X, y)
        end
        fX = classifier.oob_decision_function_
        i_finite = [ all(isfinite.(x)) for x in eachrow(fX) ]
        fX = fX[i_finite,:]
        y = y[i_finite]
        if m.is_probabilistic
            quantifier = QuaPy.Methods.PACC(m.classifier)
            quantifier.__object.pcc = QuaPy.__QUAPY.method.aggregative.PCC(classifier)
            quantifier.__object.Pte_cond_estim_ = py"paccPteCondEstim"(ScikitLearnBase.get_classes(classifier), y, fX)
            return _FittedQuaPyMethod(quantifier)
        else
            quantifier = QuaPy.Methods.ACC(m.classifier)
            y_ = mapslices(argmax, fX; dims=2)[:]
            quantifier.__object.cc = QuaPy.__QUAPY.method.aggregative.CC(classifier)
            quantifier.__object.Pte_cond_estim_ = py"accPteCondEstim"(ScikitLearnBase.get_classes(classifier), y, y_)
            return _FittedQuaPyMethod(quantifier)
        end
    elseif m.fit_classifier
        quantifier = (m.is_probabilistic ? QuaPy.Methods.PACC : QuaPy.Methods.ACC)(m.classifier)
        QuaPy.fit!(quantifier, X, y; fit_learner=m.fit_classifier)
        return _FittedQuaPyMethod(quantifier)
    else
        error("At least one of [oob_score, fit_classifier] must be true")
    end
end


# qunfold Python ACC / PACC (https://github.com/mirkobunse/qunfold)

struct _PythonACC <: QUnfold.AbstractMethod
    classifier::Any
    is_probabilistic::Bool
    solver::String
    solver_options::Union{Nothing,Dict{String,Any}}
    fit_classifier::Bool
end
PythonACC(c::Any; solver::String="trust-exact", solver_options::Union{Nothing,Dict{String,Any}}=nothing, fit_classifier::Bool=true) =
    _PythonACC(c, false, solver, solver_options, fit_classifier)
PythonPACC(c::Any; solver::String="trust-exact", solver_options::Union{Nothing,Dict{String,Any}}=nothing, fit_classifier::Bool=true) =
    _PythonACC(c, true, solver, solver_options, fit_classifier)

struct _FittedPythonACC
    quantifier::PyObject
end

qunfold = pyimport_e("qunfold")
if ispynull(qunfold) # need to install qunfold?
    Conda.pip_interop(true)
    Conda.pip("install", "qunfold @ git+https://github.com/mirkobunse/qunfold")
    Conda.pip("install", "jax[cpu]")
    qunfold = pyimport("qunfold")
end

function QUnfold.fit(m::_PythonACC, X::Any, y::AbstractVector{T}) where {T <: Integer}
    quantifier = if m.is_probabilistic
        qunfold.PACC(m.classifier, solver=m.solver, solver_options=m.solver_options, fit_classifier=m.fit_classifier)
    else
        qunfold.ACC(m.classifier, solver=m.solver, solver_options=m.solver_options, fit_classifier=m.fit_classifier)
    end
    return _FittedPythonACC(quantifier.fit(X, y))
end
QUnfold.predict(m::_FittedPythonACC, X::Any) = m.quantifier.predict(X)


# evaluation (https://github.com/HLT-ISTI/LeQua2022_scripts/blob/main/evaluate.py#L46)

_smooth(p::Vector{Float64}, ϵ::Float64) = (p .+ ϵ) ./ (1 + ϵ * length(p))
_rae(p_true::Vector{Float64}, p_hat::Vector{Float64}, ϵ::Float64) =
    mean(abs.(_smooth(p_true, ϵ) - _smooth(p_hat, ϵ)) ./ _smooth(p_true, ϵ))


# experiment

function main(;
        output_path :: String = "",
        configuration :: Symbol = :lq22,
        is_validation_run :: Bool = false,
        n_samples :: Int = -1,
        is_test_run :: Bool = false,
        )
    X_trn, y_trn = read_trn()
    n_classes = length(unique(y_trn))
    prevalence_path = "data/T1B/public/" * (is_validation_run ? "dev" : "test") * "_prevalences.txt"
    sample_dir = "data/T1B/public/" * (is_validation_run ? "dev" : "test") * "_samples"
    @info "Reading target data" prevalence_path sample_dir
    p_true = read_prevalences(prevalence_path)
    df = DataFrame(;
        sample_index = Int[],
        method = String[],
        C = Float64[],
        ae = Float64[], # absolute error
        rae = Float64[], # relative absolute error
        is_pdf = Bool[], # is the estimate a valid probability density?
        exception = String[],
    )

    # configure the experiment
    C_values = [ 0.001, 0.01, 0.1, 1.0, 10.0 ]
    n_estimators = 100
    if n_samples < 0
        n_samples = is_validation_run ? 1000 : 5000 # default values
    end
    if is_test_run
        @warn "This is a test run; results are not meaningful"
        C_values = [ 0.01, 0.1 ]
        n_estimators = 3
    end

    # grid search for the BaggingClassifier
    C_seeds = rand(UInt32, length(C_values))
    for (C, C_seed) ∈ zip(C_values, C_seeds)
        c = BaggingClassifier(
            LogisticRegression(C=C),
            n_estimators;
            oob_score = true,
            random_state = C_seed,
            n_jobs = -1
        )
        ScikitLearn.fit!(c, X_trn, y_trn) # fit the base classifier
        method_configurations = if configuration == :lq22
            [ # QuaPy methods are included for reference, softmax is :softmax_full_reg
                "EMQ (QuaPy)" => QuaPyEMQ(c; fit_classifier=false),
                "ACC (constrained)" => ACC(c; strategy=:constrained, fit_classifier=false),
                "ACC (softmax)" => ACC(c; strategy=:softmax_full_reg, fit_classifier=false),
                "ACC (pinv)" => ACC(c; strategy=:pinv, fit_classifier=false),
                "ACC (inv)" => ACC(c; strategy=:inv, fit_classifier=false),
                "ACC (QuaPy)" => QuaPyACC(c; fit_classifier=false),
                "ACC (ovr)" => ACC(c; strategy=:ovr, fit_classifier=false),
                "CC" => CC(c; fit_classifier=false),
                "PACC (constrained)" => PACC(c; strategy=:constrained, fit_classifier=false),
                "PACC (softmax)" => PACC(c; strategy=:softmax_full_reg, fit_classifier=false),
                "PACC (pinv)" => PACC(c; strategy=:pinv, fit_classifier=false),
                "PACC (inv)" => PACC(c; strategy=:inv, fit_classifier=false),
                "PACC (QuaPy)" => QuaPyPACC(c; fit_classifier=false),
                "PACC (ovr)" => PACC(c; strategy=:ovr, fit_classifier=false),
                "PCC" => PCC(c; fit_classifier=false),
            ]
        elseif configuration == :dev
            [ # constrained strategy included for reference
                "ACC (constrained)" => ACC(c; strategy=:constrained, fit_classifier=false),
                "ACC (softmax)" => ACC(c; strategy=:softmax, fit_classifier=false),
                "ACC (softmax; Python)" => PythonACC(c; fit_classifier=false),
                "ACC (softmax reg.)" => ACC(c; strategy=:softmax_reg, fit_classifier=false),
                "ACC (softmax full reg.)" => ACC(c; strategy=:softmax_full_reg, fit_classifier=false),
                "PACC (constrained)" => PACC(c; strategy=:constrained, fit_classifier=false),
                "PACC (softmax)" => PACC(c; strategy=:softmax, fit_classifier=false),
                "PACC (softmax; Python)" => PythonPACC(c; fit_classifier=false),
                "PACC (softmax reg.)" => PACC(c; strategy=:softmax_reg, fit_classifier=false),
                "PACC (softmax full reg.)" => PACC(c; strategy=:softmax_full_reg, fit_classifier=false),
            ]
        elseif configuration == :py
            [
                "ACC (solver=\"Newton-CG\")" =>
                    PythonACC(c; solver="Newton-CG", fit_classifier=false),
                "ACC (solver=\"dogleg\")" =>
                    PythonACC(c; solver="dogleg", fit_classifier=false),
                "ACC (solver=\"trust-ncg\")" =>
                    PythonACC(c;
                        solver = "trust-ncg",
                        solver_options = Dict{String,Any}("gtol" => 1e-8, "maxiter" => 1000),
                        fit_classifier = false
                    ),
                "ACC (solver=\"trust-krylov\")" =>
                    PythonACC(c; solver="trust-krylov", fit_classifier=false),
                "ACC (solver=\"trust-exact\")" =>
                    PythonACC(c; solver="trust-exact", fit_classifier=false),
                "PACC (solver=\"Newton-CG\")" =>
                    PythonPACC(c; solver="Newton-CG", fit_classifier=false),
                "PACC (solver=\"dogleg\")" =>
                    PythonPACC(c; solver="dogleg", fit_classifier=false),
                "PACC (solver=\"trust-ncg\")" =>
                    PythonPACC(c;
                        solver = "trust-ncg",
                        solver_options = Dict{String,Any}("gtol" => 1e-8, "maxiter" => 1000),
                        fit_classifier = false
                    ),
                "PACC (solver=\"trust-krylov\")" =>
                    PythonPACC(c; solver="trust-krylov", fit_classifier=false),
                "PACC (solver=\"trust-exact\")" =>
                    PythonPACC(c; solver="trust-exact", fit_classifier=false),
            ]
        else
            error("There is no configuration \"$(configuration)\"")
        end
        @info "Evaluating $(length(method_configurations)) methods" first.(method_configurations)
        methods = []
        for (method_name, method) ∈ method_configurations # fit all methods
            push!(methods, method_name => QUnfold.fit(method, X_trn, y_trn))
        end

        # evaluate the methods on all samples
        for sample_index ∈ 0:(n_samples-1)
            println("C=$(C); predicting sample $(sample_index+1)/$(n_samples)")
            X_dev = read_sample(sample_index, sample_dir)
            for (method_name, method) ∈ methods
                outcome = [sample_index, method_name, C]
                try
                    p_hat = QUnfold.predict(method, X_dev)
                    ae = mean(abs.(p_true[sample_index+1,:] - p_hat))
                    rae = _rae(p_true[sample_index+1,:], p_hat, 1/(2*size(X_dev,1)))
                    is_pdf = sum(p_hat) ≈ 1 && all(p_hat .> -sqrt(eps(Float64)))
                    push!(outcome, ae, rae, is_pdf, "")
                catch err
                    if isa(err, QUnfold.NonOptimalStatusError)
                        push!(outcome, NaN, NaN, false, string(err.termination_status))
                    elseif isa(err, SingularException)
                        push!(outcome, NaN, NaN, false, "SingularException")
                    else
                        rethrow()
                    end
                end
                push!(df, outcome)
            end
        end
    end

    # aggregations: performance metrics and failures
    df = outerjoin(
        combine( # average performance metrics
            groupby(df[df[!,:exception].=="",:], [:method, :C]),
            :ae => DataFrames.mean => :ae,
            :rae => DataFrames.mean => :rae,
            :is_pdf => (x -> DataFrames.sum(.!(x))) => :n_pdf_failures
        ),
        combine( # number of failures
            groupby(df, [:method, :C]),
            :exception => (x -> sum(x .!= "")) => :n_failures
        ),
        on = [:method, :C]
    )
    best = combine(
        groupby(df, :method),
        sdf -> begin
            sdf2 = sdf[.!(ismissing.(sdf[!,:rae])),:]
            nrow(sdf2) > 0 ? sdf2[argmin(sdf2[!,:rae]),:] : sdf[1,:]
        end
    )
    @info "Best RAE methods after hyper-parameter optimization" best

    # file export
    if length(output_path) > 0
        mkpath(dirname(output_path))
        CSV.write(output_path, df)
        @info "$(nrow(df)) results written to $(output_path)"
    end
    return df
end


# command line interface

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--configuration", "-c"
            help = "which methods to use; either lq22 (default), dev, or py"
            arg_type = Symbol
            default = :lq22
        "--is_validation_run", "-v"
            help = "whether this run should work on the validation set"
            action = :store_true
        "--n_samples", "-n"
            help = "the number of samples"
            arg_type = Int
            default = -1
        "--is_test_run", "-t"
            help = "whether this run is shortened for testing purposes"
            action = :store_true
        "output_path"
            help = "the output path for all hyper-parameter configurations"
            required = true
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; parse_commandline()...)
end
