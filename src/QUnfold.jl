module QUnfold

using JuMP
import Ipopt



# general API

abstract type Method end

fit(m::Method, X::Any, y::AbstractVector{T}) where T <: Integer =
    throw(NotImplementedError("fit not implemented for type $(typeof(m))"))



# ACC

struct ACC <: Method
    classifier::Any
end

function fit(m::ACC, X::Any, y::AbstractVector{T}) where T <: Integer
    C = length(unique(y)) # number of classes
    F = C # number of transformed features
    M = zeros(C, F) # TODO
    model = Model(Ipopt.Optimizer)
    @NLparameter(model, q[i = 1:F] == 1/F) # free parameter

    # softmax version
    @variable(model, l[1:C]) # latent variables
    @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # softmax
    @NLobjective(model, Min, sum((q[i] - sum(M[i, j] * p[j] for j in 1:C))^2 for i in 1:F))

    # constrained version
    @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
    @constraint(model, ones(C)' * p == 1) # 1' * p = 1
    @objective(model, Min, (q - M * p)' * (q - M * p))

    # solve
    for i = 1:F
        set_value(q[i], _q[i])
    end
    optimize!(model) # TODO check for convergence
    return exp.(value.(l)) ./ sum(exp.(value.(l))) # softmax version
    return value.(p) # constrained version
end



end # module
