import Ipopt, MathOptInterface.TerminationStatusCode

# utilities

struct NonOptimalStatusError <: Exception
    termination_status::TerminationStatusCode
end
Base.showerror(io::IO, x::NonOptimalStatusError) =
    print(io, "NonOptimalStatusError(", x.termination_status, ")")

function _check_termination_status(model::Model, loss::Symbol, strategy::Symbol)
    status = termination_status(model)
    if status == INTERRUPTED
        throw(InterruptException())
    elseif status ∉ [LOCALLY_SOLVED, OPTIMAL, ALMOST_LOCALLY_SOLVED, ALMOST_OPTIMAL, NUMERICAL_ERROR]
        @error "Non-optimal status after optimization" loss strategy status
        throw(NonOptimalStatusError(status))
    end
end

function _check_solver_args(M::Matrix{Float64}, q::Vector{Float64})
    if size(M, 1) != length(q)
        throw(ArgumentError("Shapes of M $(size(M)) and q ($(length(q))) do not match"))
    elseif !all(isfinite.(M))
        throw(ArgumentError("Not all values in M are finite"))
    elseif !all(isfinite.(q))
        throw(ArgumentError("Not all values in q are finite"))
    end
end


# solvers

function solve_least_squares(M::Matrix{Float64}, q::Vector{Float64}, N::Int; w::Vector{Float64}=ones(length(q)), τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy::Symbol=:constrained, λ::Float64=1e-6)
    _check_solver_args(M, q)
    if !all(isfinite.(w))
        throw(ArgumentError("Not all values in w are finite"))
    end
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    F, C = size(M) # the numbers of features and classes
    T = LinearAlgebra.diagm( # the Tikhonov matrix for optional curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]

    # least squares ||q - M*p||_2^2
    if strategy == :softmax
        @variable(model, l[1:C]) # latent variables (unconstrained)
        @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # p = softmax(l)
        if length(a) > 0
            @NLexpression(model, p_reg[i = 1:C], log10(1 + a[i] * p[i] * (N-C) / sum(a[j]*p[j] for j in 1:C)))
        else
            @NLexpression(model, p_reg[i = 1:C], p[i]) # just regularize 1/2*(Tp)^2
        end
        @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p_reg[j] for j in 1:C))
        @NLobjective(model, Min,
            sum(((q[i] - sum(M[i, j] * p[j] for j in 1:C)) / w[i])^2 for i in 1:F) # loss function
            + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
            + λ * sum(l[j]^2 for j in 1:C) # soft-max regularization
        )
    elseif strategy == :constrained && length(a) == 0 # quadratic objective without NL prefix
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @constraint(model, ones(C)' * p == 1) # 1' * p = 1
        @expression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p[j] for j in 1:C))
        @objective(model, Min,
            sum(((q[i] - sum(M[i, j] * p[j] for j in 1:C)) / w[i])^2 for i in 1:F) # loss function
            + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
        )
    elseif strategy == :constrained # NL prefix needed
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @NLconstraint(model, sum(p[i] for i in 1:C) == 1) # 1' * p = 1
        @NLexpression(model, p_reg[i = 1:C], log10(1 + a[i] * p[i] * (N-C) / sum(a[j]*p[j] for j in 1:C)))
        @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p_reg[j] for j in 1:C))
        @NLobjective(model, Min,
            sum(((q[i] - sum(M[i, j] * p[j] for j in 1:C)) / w[i])^2 for i in 1:F) # loss function
            + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
        )
    else
        error("There is no strategy \"$(strategy)\"")
    end

    # solve and return
    optimize!(model)
    _check_termination_status(model, :least_squares, strategy)
    if strategy == :softmax
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy == :constrained
        return value.(p)
    end
end


function solve_maximum_likelihood(M::Matrix{Float64}, q::Vector{Float64}, N::Int; τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy::Symbol=:constrained, n_df::Int=size(M, 2), λ::Float64=1e-6)
    _check_solver_args(M, q)
    F, C = size(M) # the numbers of features and classes
    T = LinearAlgebra.diagm( # the Tikhonov matrix for curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]
    q *= N # transform q to ̄q

    if strategy == :original # here, we assume p contains counts, not probabilities
        if length(a) == 0
            a = ones(C)
        end
        p_est = zeros(C) # the estimate of p
        diff = DiffResults.HessianResult(p_est)

        # first estimate: un-regularized least squares (Eq. 2.38 in blobel1985unfolding)
        diff = ForwardDiff.hessian!(diff, p -> sum((q - M*p).^2 ./ q) / 2, p_est)
        p_est += - inv(DiffResults.hessian(diff)) * DiffResults.gradient(diff)

        # all subsequent estimates optimize a regularized maximum likelihood
        previous_loss = Inf
        for _ in 2:100
            diff = ForwardDiff.hessian!(diff, p -> sum(M*p - q .* log.(1 .+ M*p)), p_est)
            loss_p = DiffResults.value(diff)
            gradient_p = DiffResults.gradient(diff)
            hessian_p = DiffResults.hessian(diff)

            # check for convergence
            if abs(previous_loss - loss_p) < 1e-6
                break
            end
            previous_loss = loss_p

            # eigendecomposition of the Hessian: hessian_p == U*D*U'
            eigen_p = eigen(hessian_p)
            U = eigen_p.vectors
            D = Matrix(Diagonal(eigen_p.values .^ (-1/2))) # D^(-1/2)

            # eigendecomposition of transformed Tikhonov matrix: T_2 == U_T*S*U_T'
            eigen_T = eigen(Symmetric( D*U' * T'*T * U*D ))

            # select τ (special case: no regularization if n_df == C)
            τ = n_df < C ? _select_τ(n_df, eigen_T.values)[1] : 0.0

            # Taking a step in the transformed problem and transforming back to the actual
            # solution is numerically difficult because the eigendecomposition introduces some
            # error. In the transformed problem, therefore only τ is chosen. The step is taken 
            # in the original problem instead of the commented-out solution.
            #
            # U_T = eigen_T.vectors
            # S   = diagm(eigen_T.values)
            # p_2 = 1/2 * inv(eye(S) + τ*S) * (U*D*U_T)' * (hessian_p * f - gradient_p)
            # p   = (U*D*U_T) * p_2
            #
            diff = ForwardDiff.hessian!(
                diff,
                p -> τ/2 * sum((T*log10.(1 .+ (N-C) .* a .* p ./ sum(a .* p))).^2),
                max.(0, p_est)
            )
            gradient_p += DiffResults.gradient(diff) # regularized gradient
            hessian_p += DiffResults.hessian(diff) # regularized Hessian
            p_est += - inv(hessian_p) * gradient_p
        end
        p_est = max.(0, p_est) # map to probabilities
        return p_est ./ sum(p_est)
    end

    # set up the solution vector p
    model = Model(Ipopt.Optimizer)
    set_silent(model)
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
    @NLexpression(model, Mp[i = 1:F], sum(M[i, j] * (1 + (N-C) * p[j]) for j in 1:C))
    if length(a) > 0
        @NLexpression(model, p_reg[i = 1:C], log10(1 + a[i] * p[i] * (N-C) / sum(a[j]*p[j] for j in 1:C)))
    else
        @NLexpression(model, p_reg[i = 1:C], p[i]) # just regularize 1/2*(Tp)^2
    end
    @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p_reg[j] for j in 1:C))
    @NLobjective(model, Min,
        sum(Mp[i] - q[i] * log(Mp[i]) for i in 1:F) # loss function
        + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
        + softmax_regularizer # optional soft-max regularization
    )

    # solve and return
    optimize!(model)
    _check_termination_status(model, :maximum_likelihood, strategy)
    if strategy == :softmax
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy ∈ [:constrained, :unconstrained]
        return value.(p)
    end
end

# brute-force search of a τ satisfying the n_df relation
function _select_τ(n_df::Number, eigvals_T::Vector{Float64}, min::Float64=-.01, max::Float64=-18.0, i::Int64=2)
    τ_candidates = 10 .^ range(min, stop=max, length=1000)
    n_df_candidates = map(τ -> sum([ 1/(1 + τ*v) for v in eigvals_T ]), τ_candidates)
    best = findmin(abs.(n_df_candidates .- n_df)) # tuple from difference and index of minimum
    best_τ = τ_candidates[best[2]]
    best_n_df = n_df_candidates[best[2]]
    diff = best[1]
    if i == 1 # recursive anchor
        return best_τ, best_n_df, [ diff ]
    else # search more closely around the best fit
        max = log10(τ_candidates[best[2] < length(τ_candidates) ? best[2] + 1 : best[2]]) # upper bound of subsequent search
        min = log10(τ_candidates[best[2] > 1            ? best[2] - 1 : best[2]]) # lower bound
        subsequent = _select_τ(n_df, eigvals_T, min, max, i-1) # result of recursion
        return subsequent[1], subsequent[2], vcat(diff, subsequent[3])
    end
end


function solve_hellinger_distance(M::Matrix{Float64}, q::Vector{Float64}, N::Int, n_bins::Int; τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy::Symbol=:constrained, λ::Float64=1e-6)
    _check_solver_args(M, q)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    F, C = size(M) # the numbers of "multi-features" and classes
    n_features = Int(F / n_bins) # the number of actual features in X
    T = LinearAlgebra.diagm( # the Tikhonov matrix for curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]

    # set up the solution vector p
    if strategy == :softmax
        @variable(model, l[1:C]) # latent variables (unconstrained)
        @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # p = softmax(l)
        @NLexpression(model, softmax_regularizer, λ * sum(l[j]^2 for j in 1:C))
    elseif strategy == :constrained
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @NLconstraint(model, sum(p[i] for i in 1:C) == 1)
        @NLexpression(model, softmax_regularizer, 0.0) # do not use soft-max regularization
    else
        error("There is no strategy \"$(strategy)\"")
    end

    # average feature-wise Hellinger distance
    @NLexpression(model, Mp[i = 1:F], sum(M[i, j] * p[j] for j in 1:C))
    @NLexpression(model, squared[i = 1:F], (sqrt(q[i]) - sqrt(Mp[i]))^2)
    @NLexpression(model, HD[i = 1:n_features], sqrt(sum((squared[j] for j in (1+(i-1)*n_bins):(i*n_bins)))))
    if length(a) > 0
        @NLexpression(model, p_reg[i = 1:C], log10(1 + a[i] * p[i] * (N-C) / sum(a[j]*p[j] for j in 1:C)))
    else
        @NLexpression(model, p_reg[i = 1:C], p[i]) # just regularize 1/2*(Tp)^2
    end
    @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p_reg[j] for j in 1:C))
    @NLobjective(model, Min,
        sum(HD[i] for i in 1:n_features) / n_features # loss function
        + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
        + softmax_regularizer # optional soft-max regularization
    )

    # solve and return
    optimize!(model)
    _check_termination_status(model, :hellinger_distance, strategy)
    if strategy == :softmax
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy == :constrained
        return value.(p)
    end
end
