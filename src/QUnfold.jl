module QUnfold

using JuMP, LinearAlgebra, StatsBase
import Ipopt

export ACC, fit



# general API

abstract type Method end

abstract type Transformer end

struct Fitted{T <: Method}
    method::T
    M::Matrix{Float64}
    f::Transformer
end

"""
    __fit_transform__(m, X, y) -> (f, i_M)

Fit the transformer `f` of the method `m` and return the indices of the training set `(X, y)`
which can be used for computing the transfer matrix `M`.
"""
__fit_transform__(m::Method, X::Any, y::AbstractVector{T}) where {T <: Integer} =
    error("__fit_transform__ not implemented for $(typeof(m))")

"""
__transform__(f, X) -> f(X)

Transform the data set `X` with the transformer `f`.
"""
__transform__(f::Transformer, X::Any) =
    error("__transform__ not implemented for $(typeof(f))")

"""
    __solve__(m, M, q)

Solve the optimization task defined by the transfer matrix `M` and the observed
histogram `q` with the QUnfold method `m`.
"""
__solve__(m::Method, M::Matrix{Float64}, q::Vector{Float64}) =
    error("__solve__ not implemented for $(typeof(m))")

"""
    fit(m, X, y) -> Fitted

Return a copy of the QUnfold method `m` that is fitted to the data set `(X, y)`.
"""
function fit(m::Method, X::Any, y::AbstractVector{T}) where {T <: Integer}
    f, i_M = __fit_transform__(m, X, y) # f(x) for x ∈ X
    fX = __transform__(f, X[i_M, :])
    fy = y[i_M]
    # C = length(unique(y)) # number of classes
    # F = C # number of transformed features
    M = zeros(C, C) # TODO from fX and fy
    return Fitted(m, M, f)
end

"""
    predict(m, X) -> Vector{Float64}

Predict the class prevalences in the data set `X` with the fitted method `m`.
"""
predict(m::Fitted, X::Any) =
    __solve__(m.method, m.M, mean(__transform__(m.f, X), dims=1))



# Classifier-based least squares methods: ACC, PACC

struct ACC <: Method
    classifier::Any
    val_split::Float64
    strategy::Symbol # ∈ {:softmax, :constrained}
    ACC(c::Any; val_split::Float64=.334, strategy::Symbol=:constrained) =
        new(c; val_split=val_split, strategy=strategy)
end

struct PACC <: Method
    classifier::Any
    val_split::Float64
    strategy::Symbol # ∈ {:softmax, :constrained}
    PACC(c::Any; val_split::Float64=.334, strategy::Symbol=:constrained) =
        new(c; val_split=val_split, strategy=strategy)
end

struct Classification <: Transformer
    classifier::Any
    is_probabilistic::Bool
end
Classification(m::ACC) = Classification(m.classifier, false)
Classification(m::PACC) = Classification(m.classifier, true)

function __fit_transform__(m::Union{ACC,PACC}, X::Any, y::AbstractVector{T}) where {T <: Integer}
    # TODO sample i_trn, i_tst
    fit!(m.classifier, X[i_trn, :], y[i_trn])
    return Classification(m), i_tst # = (f, i_M)
end

__transform__(f::Classification, X::Any) =
    if f.is_probabilistic
        return predict(f.classifier, X) # TODO one-hot encoding
    else
        return predict_proba(f.classifier, X)
    end

__solve__(m::Union{ACC,PACC,ReadMe}, M::Matrix{Float64}, q::Vector{Float64}) =
    solve_least_squares(M, q, m.strategy)

function solve_least_squares(M::Matrix, q::Vector{Float64}, strategy::Symbol)
    return inv(M) * q # TODO surround with try-catch
    model = Model(Ipopt.Optimizer)

    # least squares ||q - M*p||_2^2
    if strategy == :softmax
        @variable(model, l[1:C]) # latent variables (unconstrained)
        @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # p = softmax(l)
        @NLobjective(model, Min, sum((q[i] - sum(M[i, j] * p[j] for j in 1:C))^2 for i in 1:F))
    elseif strategy == :constrained
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @constraint(model, ones(C)' * p == 1) # 1' * p = 1
        @objective(model, Min, (q - M * p)' * (q - M * p))
    end

    optimize!(model) # TODO check for convergence

    if strategy == :softmax
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy == :constrained
        return value.(p)
    end
end



end # module
