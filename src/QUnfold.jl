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
    M = zeros(size(fX, 2), _n_classes(f)) # (n_features, n_classes)
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
    strategy::Symbol # ∈ {:constrained, :softmax, :softmax_reg, :softmax_full_reg, :pinv, :inv, :ovr, :none}
    is_probabilistic::Bool
    τ::Float64 # regularization strength for o-ACC and o-PACC
    a::Vector{Float64} # acceptance factors for regularization
    fit_classifier::Bool
end

"""
    ACC(classifier; kwargs...)

The Adjusted Classify & Count method, which solves a least squares objective with crisp classifier predictions.

A regularization strength `τ > 0` yields the o-ACC method for ordinal quantification, which is proposed by Bunse et al., 2022: *Ordinal Quantification through Regularization*.

**Keyword arguments**

- `strategy = :softmax` is the solution strategy (see below).
- `τ = 0.0` is the regularization strength for o-ACC.
- `a = Float64[]` are the acceptance factors for unfolding analyses.
- `fit_classifier = true` whether or not to fit the given `classifier`.

**Strategies**

For binary classification, ACC is proposed by Forman, 2008: *Quantifying counts and costs via classification*. In the multi-class setting, multiple extensions are available.

- `:softmax` (default; our method) improves `:softmax_full_reg` by setting one latent parameter to zero instead of introducing a technical regularization term.
- `:constrained` constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: *A method of automated nonparametric content analysis for social science*.
- `:pinv` computes a pseudo-inverse akin to a minimum-norm constraint, as discussed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:inv` computes the true inverse (if existent) of the transfer matrix `M`, as proposed by Vucetic & Obradovic, 2001: *Classification on data with biased class distribution*.
- `:ovr` solves multiple binary one-versus-rest adjustments, as proposed by Forman (2008).
- `:none` yields the `CC` method without any adjustment.
- `:softmax_full_reg` (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:softmax_reg` (our method) is a variant of `:softmax`, which sets one latent parameter to zero in addition to introducing a technical regularization term.
"""
ACC(c::Any; strategy::Symbol=:softmax, τ::Float64=0.0, a::Vector{Float64}=Float64[], fit_classifier::Bool=true) =
    _ACC(c, strategy, false, τ, a, fit_classifier)

"""
    PACC(classifier; kwargs...)

The Probabilistic Adjusted Classify & Count method, which solves a least squares objective with predictions of posterior probabilities.

A regularization strength `τ > 0` yields the o-PACC method for ordinal quantification, which is proposed by Bunse et al., 2022: *Ordinal Quantification through Regularization*.

**Keyword arguments**

- `strategy = :softmax` is the solution strategy (see below).
- `τ = 0.0` is the regularization strength for o-PACC.
- `a = Float64[]` are the acceptance factors for unfolding analyses.
- `fit_classifier = true` whether or not to fit the given `classifier`.

**Strategies**

For binary classification, PACC is proposed by Bella et al., 2010: *Quantification via Probability Estimators*. In the multi-class setting, multiple extensions are available.

- `:softmax` (default; our method) improves `:softmax_full_reg` by setting one latent parameter to zero instead of introducing a technical regularization term.
- `:constrained` constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: *A method of automated nonparametric content analysis for social science*.
- `:pinv` computes a pseudo-inverse akin to a minimum-norm constraint, as discussed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:inv` computes the true inverse (if existent) of the transfer matrix `M`, as proposed by Vucetic & Obradovic, 2001: *Classification on data with biased class distribution*.
- `:ovr` solves multiple binary one-versus-rest adjustments, as proposed by Forman (2008).
- `:none` yields the `CC` method without any adjustment.
- `:softmax_full_reg` (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:softmax_reg` (our method) is a variant of `:softmax`, which sets one latent parameter to zero in addition to introducing a technical regularization term.
"""
PACC(c::Any; strategy::Symbol=:softmax, τ::Float64=0.0, a::Vector{Float64}=Float64[], fit_classifier::Bool=true) =
    _ACC(c, strategy, true, τ, a, fit_classifier)

"""
    CC(classifier; kwargs...)

The Classify & Count method, which uses crisp classifier predictions without any adjustment. This weak baseline method is proposed by Forman, 2008: *Quantifying counts and costs via classification*.

**Keyword arguments**

- `fit_classifier = true` whether or not to fit the given `classifier`.
"""
CC(c::Any; fit_classifier::Bool=true) =
    _ACC(c, :none, false, 0.0, Float64[], fit_classifier)

"""
    PCC(classifier; kwargs...)

The Probabilistic Classify & Countmethod, which uses predictions of posterior probabilities without any adjustment. This method is proposed by Bella et al., 2010: *Quantification via Probability Estimators*.

**Keyword arguments**

- `fit_classifier = true` whether or not to fit the given `classifier`.
"""
PCC(c::Any; fit_classifier::Bool=true) =
    _ACC(c, :none, true, 0.0, Float64[], fit_classifier)

_transformer(m::_ACC) = ClassTransformer(
        m.classifier;
        is_probabilistic = m.is_probabilistic,
        fit_classifier = m.fit_classifier
    )

_solve(m::_ACC, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int) =
    if m.strategy ∈ [:constrained, :softmax, :softmax_reg, :softmax_full_reg]
        solve_least_squares(M, q, N; τ=m.τ, a=m.a, strategy=m.strategy)
    elseif m.strategy == :pinv
        if any(sum(M; dims=2) .== 0) # limit the estimation to non-zero features
            nonzero = sum(M; dims=2)[:] .> 0
            q = q[nonzero]
            M = M[nonzero, :]
        end
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
    strategy::Symbol # ∈ {:constrained, :softmax, :softmax_reg, :softmax_full_reg, :unconstrained}
    initialization::Symbol # ∈ {:random, :uniform, :training, :blobel}
end

"""
    RUN(transformer; kwargs...)

The Regularized Unfolding method by Blobel, 1985: *Unfolding methods in high-energy physics experiments*.

**Keyword arguments**

- `strategy = :softmax` is the solution strategy (see below).
- `τ = 1e-6` is the regularization strength for ordinal quantification.
- `n_df = -1` (only used if `strategy==:original`) is the effective number of degrees of freedom, required to be `0 < n_df <= C` where `C` is the number of classes.
- `a = Float64[]` are the acceptance factors for unfolding analyses.

**Strategies**

Blobel's loss function, feature transformation, and regularization can be optimized with multiple strategies.

- `:softmax` (default; our method) improves `:softmax_full_reg` by setting one latent parameter to zero instead of introducing a technical regularization term.
- `:original` is the original, unconstrained Newton optimization proposed by Blobel (1985).
- `:constrained` constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: *A method of automated nonparametric content analysis for social science*.
- `:softmax_full_reg` (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:softmax_reg` (our method) is a variant of `:softmax`, which sets one latent parameter to zero in addition to introducing a technical regularization term.
- `:unconstrained` (our method) is similar to `:original`, but uses a more generic solver.
"""
RUN(transformer::Union{AbstractTransformer,FittedTransformer}; τ::Float64=1e-6, n_df::Int=-1, a::Vector{Float64}=Float64[], strategy=:softmax, initialization=:random) =
    _RUN_SVD(transformer, :run, τ, n_df, a, strategy, initialization)

"""
    SVD(transformer; kwargs...)

The The Singular Value Decomposition-based unfolding method by Hoecker & Kartvelishvili, 1996: *SVD approach to data unfolding*.

**Keyword arguments**

- `strategy = :softmax` is the solution strategy (see below).
- `τ = 1e-6` is the regularization strength for ordinal quantification.
- `n_df = -1` (only used if `strategy==:original`) is the effective rank, required to be `0 < n_df < C` where `C` is the number of classes.
- `a = Float64[]` are the acceptance factors for unfolding analyses.

**Strategies**

Hoecker & Kartvelishvili's loss function, feature transformation, and regularization can be optimized with multiple strategies.

- `:softmax` (default; our method) improves `:softmax_full_reg` by setting one latent parameter to zero instead of introducing a technical regularization term.
- `:original` is the original, analytic solution proposed by Hoecker & Kartvelishvili (1996).
- `:constrained` constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: *A method of automated nonparametric content analysis for social science*.
- `:softmax_full_reg` (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:softmax_reg` (our method) is a variant of `:softmax`, which sets one latent parameter to zero in addition to introducing a technical regularization term.
- `:unconstrained` (our method) is similar to `:original`, but uses a more generic solver.
"""
SVD(transformer::Union{AbstractTransformer,FittedTransformer}; τ::Float64=1e-6, n_df::Int=-1, a::Vector{Float64}=Float64[], strategy=:softmax) =
    _RUN_SVD(transformer, :svd, τ, n_df, a, strategy, :random)
_transformer(m::_RUN_SVD) = m.transformer
_solve(m::_RUN_SVD, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int, b::Vector{Float64}=zeros(length(q))) =
    if m.loss == :run
        solve_maximum_likelihood(M, q, N, p_trn, b; τ=m.τ, n_df=m.n_df > 0 ? m.n_df : size(M, 2), a=m.a, strategy=m.strategy, initialization=m.initialization)
    elseif m.loss == :svd # weighted least squares
        strategy = m.strategy == :original ? :svd : m.strategy # rename :original -> :svd
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

"""
    HDx(n_bins; kwargs...)

The Hellinger Distance-based method on feature histograms by González-Castro et al., 2013: *Class distribution estimation based on the Hellinger distance*.

The parameter `n_bins` specifies the number of bins *per feature*. A regularization strength `τ > 0` yields the o-HDx method for ordinal quantification, which is proposed by Bunse et al., 2022: *Machine learning for acquiring knowledge in astro-particle physics*.

**Keyword arguments**

- `strategy = :softmax` is the solution strategy (see below).
- `τ = 0.0` is the regularization strength for o-HDx.
- `a = Float64[]` are the acceptance factors for unfolding analyses.

**Strategies**

González-Castro et al.'s loss function and feature transformation can be optimized with multiple strategies.

- `:softmax` (default; our method) improves `:softmax_full_reg` by setting one latent parameter to zero instead of introducing a technical regularization term.
- `:constrained` constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: *A method of automated nonparametric content analysis for social science*.
- `:softmax_full_reg` (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:softmax_reg` (our method) is a variant of `:softmax`, which sets one latent parameter to zero in addition to introducing a technical regularization term.
"""
struct HDx <: AbstractMethod
    n_bins::Int
    τ::Float64 # regularization strength
    a::Vector{Float64} # acceptance factors for regularization
    strategy::Symbol # ∈ {:constrained, :softmax, :softmax_reg, :softmax_full_reg}
    HDx(n_bins::Int; τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy=:softmax) = new(n_bins, τ, a, strategy)
end

"""
    HDy(classifier, n_bins; kwargs...)

The Hellinger Distance-based method on prediction histograms by González-Castro et al., 2013: *Class distribution estimation based on the Hellinger distance*.

The parameter `n_bins` specifies the number of bins *per class*. A regularization strength `τ > 0` yields the o-HDx method for ordinal quantification, which is proposed by Bunse et al., 2022: *Machine learning for acquiring knowledge in astro-particle physics*.

**Keyword arguments**

- `strategy = :softmax` is the solution strategy (see below).
- `τ = 0.0` is the regularization strength for o-HDx.
- `a = Float64[]` are the acceptance factors for unfolding analyses.
- `fit_classifier = true` whether or not to fit the given `classifier`.

**Strategies**

González-Castro et al.'s loss function and feature transformation can be optimized with multiple strategies.

- `:softmax` (default; our method) improves `:softmax_full_reg` by setting one latent parameter to zero instead of introducing a technical regularization term.
- `:constrained` constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: *A method of automated nonparametric content analysis for social science*.
- `:softmax_full_reg` (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: *On Multi-Class Extensions of Adjusted Classify and Count*.
- `:softmax_reg` (our method) is a variant of `:softmax`, which sets one latent parameter to zero in addition to introducing a technical regularization term.
"""
struct HDy <: AbstractMethod
    classifier::Any
    n_bins::Int
    τ::Float64 # regularization strength
    a::Vector{Float64} # acceptance factors for regularization
    strategy::Symbol # ∈ {:constrained, :softmax, :softmax_reg, :softmax_full_reg}
    fit_classifier::Bool
    HDy(classifier::Any, n_bins::Int; τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy=:softmax, fit_classifier::Bool=true) =
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

"""
    IBU(transformer, n_bins; kwargs...)

The Iterative Bayesian Unfolding method by D'Agostini, 1995: *A multidimensional unfolding method based on Bayes' theorem*.

**Keyword arguments**

- `o = 0` is the order of the polynomial for ordinal quantification.
- `λ = 0.0` is the impact of the polynomial for ordinal quantification.
- `a = Float64[]` are the acceptance factors for unfolding analyses.
"""
struct IBU <: AbstractMethod
    transformer::Union{AbstractTransformer,FittedTransformer}
    o::Int # order of the polynomial
    λ::Float64 # impact of the polynomial
    a::Vector{Float64} # acceptance factors for regularization
    n_iterations::Int # maximum number of interations
    ϵ::Float64 # minimum Chi-Square distance between consecutive estimates
    IBU(transformer::Union{AbstractTransformer,FittedTransformer}; o::Int=-1, λ::Float64=.0, a::Vector{Float64}=Float64[], n_iterations::Int=100, ϵ::Float64=.0) =
        new(transformer, o, λ, a, n_iterations, ϵ)
end
_transformer(m::IBU) = m.transformer
_solve(m::IBU, M::Matrix{Float64}, q::Vector{Float64}, p_trn::Vector{Float64}, N::Int) =
    solve_expectation_maximization(M, q, N, ones(size(M, 2)) ./ size(M, 2); o=m.o, λ=m.λ, a=m.a, n_iterations=m.n_iterations, ϵ=m.ϵ)

"""
    SLD(classifier; kwargs...)

The Saerens-Latinne-Decaestecker method, a.k.a. EMQ or Expectation Maximization-based Quantification by Saerens et al., 2002: *Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure*.

A polynomial order `o > 0` and regularization impact `λ > 0` yield the o-SLD method for ordinal quantification, which is proposed by Bunse et al., 2022: *Machine learning for acquiring knowledge in astro-particle physics*.

**Keyword arguments**

- `o = 0` is the order of the polynomial for o-SLD.
- `λ = 0.0` is the impact of the polynomial for o-SLD.
- `a = Float64[]` are the acceptance factors for unfolding analyses.
- `fit_classifier = true` whether or not to fit the given `classifier`.
"""
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
    f = _fit_transform(t, X, y)[1] # ensures that minimum(y) ∈ [0, 1]
    p_trn = [ mean((y .+ (1 - minimum(y))) .== i) for i ∈ 1:_n_classes(f) ]
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
