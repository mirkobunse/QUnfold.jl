module QUnfold

using LinearAlgebra, StatsBase

export fit, predict, ACC, PACC

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


# classification-based least squares methods: ACC, PACC

struct _ACC <: AbstractMethod
    classifier::Any
    strategy::Symbol # ∈ {:softmax, :constrained}
    is_probabilistic::Bool
end
ACC(c::Any; strategy::Symbol=:constrained) = _ACC(c, strategy, false)
PACC(c::Any; strategy::Symbol=:constrained) = _ACC(c, strategy, true)

_transformer(m::_ACC) = ClassTransformer(m.classifier, m.is_probabilistic)

_solve(m::_ACC, M::Matrix{Float64}, q::Vector{Float64}) =
    if m.strategy ∈ [:constrained, :softmax]
        solve_least_squares(M, q, m.strategy)
    elseif m.strategy == :pinv
        pinv(M) * q
    elseif m.strategy == :inv
        inv(M) * q
    else
        error("There is no strategy \"$(m.strategy)\"")
    end


end # module
