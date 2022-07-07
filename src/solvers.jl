using JuMP
import Ipopt, MathOptInterface.TerminationStatusCode

struct NonOptimalStatusError <: Exception
    termination_status::TerminationStatusCode
end
Base.showerror(io::IO, x::NonOptimalStatusError) =
    print(io, "NonOptimalStatusError(", x.termination_status, ")")

function solve_least_squares(M::Matrix, q::Vector{Float64}; w::Vector{Float64}=ones(length(q)), τ::Float64=0.0, strategy::Symbol=:constrained, λ::Float64=1e-6)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    F, C = size(M) # the numbers of classes and features
    T = LinearAlgebra.diagm( # the Tikhonov matrix for optional curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]

    # least squares ||q - M*p||_2^2
    if strategy == :softmax
        @variable(model, l[1:C]) # latent variables (unconstrained)
        @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # p = softmax(l)
        @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p[j] for j in 1:C))
        @NLobjective(model, Min,
            sum(((q[i] - sum(M[i, j] * p[j] for j in 1:C)) / w[i])^2 for i in 1:F) # loss function
            + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
            + λ * sum(l[j]^2 for j in 1:C) # soft-max regularization
        )
    elseif strategy == :constrained
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @constraint(model, ones(C)' * p == 1) # 1' * p = 1
        @expression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p[j] for j in 1:C))
        @objective(model, Min,
            ((q - M * p) ./ w)' * ((q - M * p) ./ w) # loss function
            + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
        )
    else
        error("There is no strategy \"$(strategy)\"")
    end

    # solve and return
    optimize!(model)
    _check_termination_status(model, strategy)
    if strategy == :softmax
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy == :constrained
        return value.(p)
    end
end

function solve_maximum_likelihood(M::Matrix, q::Vector{Float64}, N::Int; τ::Float64=0.0, strategy::Symbol=:constrained, λ::Float64=1e-6)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    F, C = size(M) # the numbers of classes and features
    T = LinearAlgebra.diagm( # the Tikhonov matrix for curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]
    q *= N # transform q to ̄q

    # set up the solution vector p
    if strategy == :softmax
        @variable(model, l[1:C]) # latent variables (unconstrained)
        @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # p = softmax(l)
        @NLexpression(model, softmax_regularizer, λ * sum(l[j]^2 for j in 1:C))
    elseif strategy == :constrained
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @NLconstraint(model, sum(p[i] for i in 1:C) == 1)
        @NLexpression(model, softmax_regularizer, 0.0) # do not use soft-max regularization
    elseif strategy == :unconstrained
        @variable(model, p[1:C])
        @NLexpression(model, softmax_regularizer, 0.0)
    else
        error("There is no strategy \"$(strategy)\"")
    end

    # minimum regularized log-likelihood  ∑_{i=1}^F [Mp]_i - q_i ln [Mp]_i  +  τ/2 * (Tp)^2
    @NLexpression(model, Mp[i = 1:F], sum(M[i, j] * N * p[j] for j in 1:C))
    @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * N * p[j] for j in 1:C))
    @NLobjective(model, Min,
        sum(Mp[i] - q[i] * log(Mp[i]) for i in 1:F) # loss function
        + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
        + softmax_regularizer # optional soft-max regularization
    )

    # solve and return
    optimize!(model)
    _check_termination_status(model, strategy)
    if strategy == :softmax
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy ∈ [:constrained, :unconstrained]
        return value.(p)
    end
end

function _check_termination_status(model::Model, strategy::Symbol)
    status = termination_status(model)
    if status == INTERRUPTED
        throw(InterruptException())
    elseif status ∉ [LOCALLY_SOLVED, OPTIMAL, ALMOST_LOCALLY_SOLVED, ALMOST_OPTIMAL]
        @error "Non-optimal status after optimization" strategy status
        throw(NonOptimalStatusError(status))
    end
end
