module QUnfold

using
    DiffResults,
    ForwardDiff,
    JuMP,
    LinearAlgebra,
    Preferences,
    Random,
    Requires,
    StatsBase

export
    ACC,
    CC,
    ClassTransformer,
    fit,
    HDx,
    HDy,
    IBU,
    PACC,
    PCC,
    predict,
    predict_with_background,
    RUN,
    SLD,
    SVD,
    TreeTransformer

include("transformers.jl")
include("solvers.jl")

# add an additional constructor TreeTransformer(n_bins::Int; ...) when PyCall is loaded
function __init__()
    set_preferences!(ForwardDiff, "nansafe_mode" => true)
    @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
        TreeTransformer(n_bins::Int; kwargs...) =
            TreeTransformer(
                PyCall.pyimport_conda("sklearn.tree", "scikit-learn").DecisionTreeClassifier(; max_leaf_nodes=n_bins);
                kwargs...
            )
    end
end


# general API

abstract type AbstractMethod end

struct FittedMethod{S<:AbstractMethod, T<:FittedTransformer}
    method::S
    M::Matrix{Float64}
    f::T
    p_trn::Vector{Float64}
end

"""
    _transformer(m)

Return the transformer of the QUnfold method `m`.
"""
_transformer(m::AbstractMethod) =
    error("_transformer not implemented for $(typeof(m))")

"""
    _solve(m, M, q, p_trn, N)

Solve the optimization task defined by the transfer matrix `M`, the observed
histogram `q`, the training prevalences `p_trn`, and the number `N` of samples
with the QUnfold method `m`.
"""
_solve(m::AbstractMethod, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int) =
    error("_solve not implemented for $(typeof(m))")

"""
    fit(m, X, y) -> FittedMethod

Return a copy of the QUnfold method `m` that is fitted to the data set `(X, y)`.
"""
function fit(m::AbstractMethod, X::Any, y::AbstractVector{T}) where {T <: Integer}
    f, fX, fy = _fit_transform(_transformer(m), X, y) # f(x) for x ∈ X
    M = zeros(size(fX, 2), length(unique(y))) # (n_features, n_classes)
    for (fX_i, fy_i) in zip(eachrow(fX), fy)
        M[:, fy_i] .+= fX_i # one histogram of f(X) per class
    end
    p_trn = sum(M; dims=1)[:] / sum(M)
    return FittedMethod(m, M ./ sum(M; dims=1), f, p_trn) # normalize M
end

"""
    predict(m, X) -> Vector{Float64}

Predict the class prevalences in the data set `X` with the fitted method `m`.
"""
predict(m::FittedMethod, X::Any) =
    _solve(m.method, m.M, mean(_transform(m.f, X), dims=1)[:], m.p_trn, size(X, 1))

"""
    predict_with_background(m, X, X_b, α=1) -> Vector{Float64}

Predict the class prevalences in the observed data set `X` with the fitted
method `m`, taking into account a background measurement `X_b` that is scaled
by `α`.
"""
predict_with_background(m::FittedMethod, X::Any, X_b::Any, α::Float64=1.) =
    _solve(m.method, m.M, mean(_transform(m.f, X), dims=1)[:], m.p_trn, size(X, 1), (α * size(X_b, 1) / size(X, 1)) .* mean(_transform(m.f, X_b), dims=1)[:])
    # _solve(m.method, m.M, mean(_transform(m.f, X), dims=1)[:], m.p_trn, size(X, 1), α, Float64.(sum(_transform(m.f, X_b), dims=1)[:]))


# utility methods

function clip_and_normalize(x::Vector{Float64})
    x[x .< 0] .= 0
    x[x .> 1] .= 1
    return x ./ sum(x)
end


# classification-based least squares and un-adjusted methods: ACC, PACC, CC, PCC

struct _ACC <: AbstractMethod
    classifier::Any
    strategy::Symbol # ∈ {:constrained, :softmax, :pinv, :inv, :ovr, :none}
    is_probabilistic::Bool
    τ::Float64 # regularization strength for o-ACC and o-PACC
    a::Vector{Float64} # acceptance factors for regularization
    fit_classifier::Bool
end
ACC(c::Any; strategy::Symbol=:constrained, τ::Float64=0.0, a::Vector{Float64}=Float64[], fit_classifier::Bool=true) =
    _ACC(c, strategy, false, τ, a, fit_classifier)
PACC(c::Any; strategy::Symbol=:constrained, τ::Float64=0.0, a::Vector{Float64}=Float64[], fit_classifier::Bool=true) =
    _ACC(c, strategy, true, τ, a, fit_classifier)
CC(c::Any; fit_classifier::Bool=true) =
    _ACC(c, :none, false, 0.0, Float64[], fit_classifier)
PCC(c::Any; fit_classifier::Bool=true) =
    _ACC(c, :none, true, 0.0, Float64[], fit_classifier)

_transformer(m::_ACC) = ClassTransformer(
        m.classifier;
        is_probabilistic = m.is_probabilistic,
        fit_classifier = m.fit_classifier
    )

_solve(m::_ACC, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int) =
    if m.strategy ∈ [:constrained, :softmax]
        solve_least_squares(M, q, N; τ=m.τ, a=m.a, strategy=m.strategy)
    elseif m.strategy == :pinv
        clip_and_normalize(pinv(M) * q)
    elseif m.strategy == :inv
        clip_and_normalize(inv(M) * q)
    elseif m.strategy == :ovr
        true_positive_rates = diag(M)
        false_positive_rates = (1 .- diag(M .* reshape(p_trn, (1, length(p_trn))))) ./ (1 .- p_trn) # renormalize M
        p = (q - false_positive_rates) ./ (true_positive_rates - false_positive_rates)
        clip_and_normalize(p)
    elseif m.strategy == :none
        q
    else
        error("There is no strategy \"$(m.strategy)\"")
    end


# RUN and SVD

struct _RUN_SVD <: AbstractMethod
    transformer::Union{AbstractTransformer,FittedTransformer}
    loss::Symbol # ∈ {:run, :svd}
    τ::Float64 # regularization strength
    n_df::Int # alternative regularization strength
    a::Vector{Float64} # acceptance factors for regularization
    strategy::Symbol # ∈ {:constrained, :softmax, :unconstrained}
end
RUN(transformer::Union{AbstractTransformer,FittedTransformer}; τ::Float64=1e-6, n_df::Int=-1, a::Vector{Float64}=Float64[], strategy=:constrained) =
    _RUN_SVD(transformer, :run, τ, n_df, a, strategy)
SVD(transformer::Union{AbstractTransformer,FittedTransformer}; τ::Float64=1e-6, n_df::Int=-1, a::Vector{Float64}=Float64[], strategy=:constrained) =
    _RUN_SVD(transformer, :svd, τ, n_df, a, strategy)
_transformer(m::_RUN_SVD) = m.transformer
_solve(m::_RUN_SVD, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int, b::Vector{Float64}=zeros(length(q))) =
    if m.loss == :run
        solve_maximum_likelihood(M, q, N, b; τ=m.τ, n_df=m.n_df > 0 ? m.n_df : size(M, 2), a=m.a, strategy=m.strategy)
    elseif m.loss == :svd # weighted least squares
        strategy = m.strategy == :original ? :svd : m.strategy
        n_df = m.n_df > 0 ? m.n_df : size(M, 2)
        solve_least_squares(M, q, N; w=_svd_weights(q, N), τ=m.τ, n_df=n_df, a=m.a, strategy=strategy)
    else
        error("There is no loss \"$(m.loss)\"")
    end
function _svd_weights(q::Vector{Float64}, N::Int)
    w = sqrt.(1 .+ (N-length(q)) .* q)
    return w ./ mean(w) # the mean weight will be 1 after this normalization
end


# HDx and HDy

struct HDx <: AbstractMethod
    n_bins::Int
    τ::Float64 # regularization strength
    a::Vector{Float64} # acceptance factors for regularization
    strategy::Symbol # ∈ {:constrained, :softmax}
    HDx(n_bins::Int; τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy=:constrained) = new(n_bins, τ, a, strategy)
end
struct HDy <: AbstractMethod
    classifier::Any
    n_bins::Int
    τ::Float64 # regularization strength
    a::Vector{Float64} # acceptance factors for regularization
    strategy::Symbol # ∈ {:constrained, :softmax}
    fit_classifier::Bool
    HDy(classifier::Any, n_bins::Int; τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy=:constrained, fit_classifier::Bool=true) =
        new(classifier, n_bins, τ, a, strategy, fit_classifier)
end
_transformer(m::HDx) = HistogramTransformer(m.n_bins)
_transformer(m::HDy) = HistogramTransformer(
        m.n_bins;
        preprocessor = ClassTransformer(
            m.classifier;
            is_probabilistic = true,
            fit_classifier = m.fit_classifier
        )
    )
_solve(m::Union{HDx,HDy}, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int, b::Vector{Float64}=zeros(length(q))) =
    solve_hellinger_distance(M, q, N, m.n_bins, b; τ=m.τ, a=m.a, strategy=m.strategy)


# IBU and SLD

struct IBU <: AbstractMethod
    transformer::Union{AbstractTransformer,FittedTransformer}
    o::Int # order of the polynomial
    λ::Float64 # impact of the polynomial
    a::Vector{Float64} # acceptance factors for regularization
    IBU(transformer::Union{AbstractTransformer,FittedTransformer}; o::Int=-1, λ::Float64=.0, a::Vector{Float64}=Float64[]) =
        new(transformer, o, λ, a)
end
_transformer(m::IBU) = m.transformer
_solve(m::IBU, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int) =
    solve_expectation_maximization(M, q, N, ones(size(M, 2)) ./ size(M, 2); o=m.o, λ=m.λ, a=m.a)

struct SLD <: AbstractMethod
    classifier::Any
    o::Int # order of the polynomial
    λ::Float64 # impact of the polynomial
    a::Vector{Float64} # acceptance factors for regularization
    fit_classifier::Bool
    SLD(classifier::Any; o::Int=-1, λ::Float64=.0, a::Vector{Float64}=Float64[], fit_classifier::Bool=true) =
        new(classifier, o, λ, a, fit_classifier)
end
function fit(m::SLD, X::Any, y::AbstractVector{T}) where {T <: Integer}
    t = ClassTransformer(m.classifier; is_probabilistic=true, fit_classifier=m.fit_classifier)
    f = _fit_transform(t, X, y)[1]
    p_trn = [ mean(y .== i) for i ∈ 1:length(unique(y)) ]
    return FittedMethod(m, Matrix{Float64}(undef, 0, 0), f, p_trn)
end
predict(m::FittedMethod{SLD,FittedClassTransformer}, X::Any) =
    solve_expectation_maximization(
        _transform(m.f, X) ./ m.p_trn', # M = h(x) / p_trn
        ones(size(X, 1)) ./ size(X, 1), # q = 1/N
        size(X, 1), # N
        m.p_trn; # p_0 = p_trn
        o = m.method.o,
        λ = m.method.λ,
        a = m.method.a
    )

end # module
