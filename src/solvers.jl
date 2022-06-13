using JuMP
import Ipopt
import MathOptInterface: TerminationStatusCode, INTERRUPTED, LOCALLY_SOLVED, OPTIMAL

struct NonOptimalStatusError <: Exception
    termination_status::TerminationStatusCode
end
Base.showerror(io::IO, x::NonOptimalStatusError) =
    print(io, "NonOptimalStatusError(", x.termination_status, ")")

function solve_least_squares(M::Matrix, q::Vector{Float64}, strategy::Symbol)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    C, F = size(M) # the numbers of classes and features

    # least squares ||q - M*p||_2^2
    if strategy == :softmax
        @variable(model, -1 ≤ l[1:C] ≤ 1) # latent variables (unconstrained sum)
        @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # p = softmax(l)
        @NLobjective(model, Min, sum((q[i] - sum(M[i, j] * p[j] for j in 1:C))^2 for i in 1:F))
    elseif strategy == :constrained
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @constraint(model, ones(C)' * p == 1) # 1' * p = 1
        @objective(model, Min, (q - M * p)' * (q - M * p))
    end

    optimize!(model)
    status = termination_status(model)
    if status == INTERRUPTED
        throw(InterruptException())
    elseif status ∉ [LOCALLY_SOLVED, OPTIMAL]
        p_est = exp.(value.(l)) ./ sum(exp.(value.(l)))
        @error "Non-optimal status after optimization" strategy status maximum(p_est) minimum(p_est) maximum(value.(l)) minimum(value.(l))
        throw(NonOptimalStatusError(status))
    end

    if strategy == :softmax
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy == :constrained
        return value.(p)
    end
end
