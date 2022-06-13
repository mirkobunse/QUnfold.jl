module QUnfold

using LinearAlgebra, StatsBase

export fit, predict, ACC, CC, PACC, PCC

include("transformers.jl")
include("solvers.jl")


# general API

abstract type AbstractMethod end

struct FittedMethod{S<:AbstractMethod, T<:FittedTransformer}
    method::S
    M::Matrix{Float64}
    f::T
end

"""
    _transformer(m)

Return the transformer of the QUnfold method `m`.
"""
_transformer(m::AbstractMethod) =
    error("_transformer not implemented for $(typeof(m))")

"""
    _solve(m, M, q)

Solve the optimization task defined by the transfer matrix `M` and the observed
histogram `q` with the QUnfold method `m`.
"""
_solve(m::AbstractMethod, M::Matrix{Float64}, q::Vector{Float64}) =
    error("_solve not implemented for $(typeof(m))")

"""
    fit(m, X, y) -> FittedMethod

Return a copy of the QUnfold method `m` that is fitted to the data set `(X, y)`.
"""
function fit(m::AbstractMethod, X::Any, y::AbstractVector{T}) where {T <: Integer}
    f, fX, fy = _fit_transform(_transformer(m), X, y) # f(x) for x ∈ X
    M = zeros(length(unique(y)), size(fX, 2)) # (n_classes, n_features)
    for (fX_i, fy_i) in zip(eachrow(fX), fy)
        M[fy_i, :] .+= fX_i # one histogram of f(X) per class
    end
    return FittedMethod(m, M ./ sum(M; dims=2), f) # normalize M
end

"""
    predict(m, X) -> Vector{Float64}

Predict the class prevalences in the data set `X` with the fitted method `m`.
"""
predict(m::FittedMethod, X::Any) =
    _solve(m.method, m.M, mean(_transform(m.f, X), dims=1)[:])


# utility methods

function clip_and_normalize(x::Vector{Float64})
    x[x .< 0] .= 0
    # x[x .> 1] .= 1
    return x ./ sum(x)
end


# classification-based least squares and un-adjusted methods: ACC, PACC, CC, PCC

struct _ACC <: AbstractMethod
    classifier::Any
    strategy::Symbol # ∈ {:constrained, :softmax, :pinv, :inv, :none}
    is_probabilistic::Bool
    fit_classifier::Bool
end
ACC(c::Any; strategy::Symbol=:constrained, fit_classifier::Bool=true) =
    _ACC(c, strategy, false, fit_classifier)
PACC(c::Any; strategy::Symbol=:constrained, fit_classifier::Bool=true) =
    _ACC(c, strategy, true, fit_classifier)
CC(c::Any; fit_classifier::Bool=true) =
    _ACC(c, :none, false, fit_classifier)
PCC(c::Any; fit_classifier::Bool=true) =
    _ACC(c, :none, true, fit_classifier)

_transformer(m::_ACC) = ClassTransformer(m.classifier, m.is_probabilistic, m.fit_classifier)

_solve(m::_ACC, M::Matrix{Float64}, q::Vector{Float64}) =
    if m.strategy ∈ [:constrained, :softmax]
        solve_least_squares(M, q, m.strategy)
    elseif m.strategy == :pinv
        clip_and_normalize(pinv(M) * q)
    elseif m.strategy == :inv
        clip_and_normalize(inv(M) * q)
    elseif m.strategy == :none
        q
    else
        error("There is no strategy \"$(m.strategy)\"")
    end


end # module
