import Ipopt, MathOptInterface.TerminationStatusCode, Polynomials

# utilities

struct NonOptimalStatusError <: Exception
    termination_status::TerminationStatusCode
end
Base.showerror(io::IO, x::NonOptimalStatusError) =
    print(io, "NonOptimalStatusError(", x.termination_status, ")")

function _check_termination_status(status::TerminationStatusCode, loss::Symbol, strategy::Symbol, M::Matrix{Float64}, q::Vector{Float64})
    if status == INTERRUPTED
        throw(InterruptException())
    elseif status ∉ [LOCALLY_SOLVED, OPTIMAL, ALMOST_LOCALLY_SOLVED, ALMOST_OPTIMAL, ITERATION_LIMIT, NUMERICAL_ERROR]
        @error "Non-optimal status after optimization" loss strategy status
        println("M = $M\nq = $q")
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

function solve_least_squares(M::Matrix{Float64}, q::Vector{Float64}, N::Int; w::Vector{Float64}=ones(length(q)), τ::Float64=0.0, n_df::Int=size(M, 2), a::Vector{Float64}=Float64[], strategy::Symbol=:softmax, λ::Float64=1e-6)
    _check_solver_args(M, q)
    if !all(isfinite.(w))
        throw(ArgumentError("Not all values in w are finite"))
    end
    if any(sum(M; dims=2) .== 0) # limit the estimation to non-zero features
        nonzero = sum(M; dims=2)[:] .> 0
        q = q[nonzero]
        M = M[nonzero, :]
        w = w[nonzero]
    end
    F, C = size(M) # the numbers of features and classes

    if strategy == :svd # here, we assume p contains counts, not probabilities
        q = 1 .+ (N-C) .* q # transform q to \bar{q}
        B = diagm(0 => q)
        inv_T = inv(diagm(-1=>ones(C-1), 0=>1e-3 .- vcat(1, 2*ones(C-2), 1), 1=>ones(C-1)))

        # re-scaling and rotation steps 1-5 (without step 3) [hoecker1995svd]
        m, Q = LinearAlgebra.eigen(B) # transformation step 1
        M_tilde = Matrix(Diagonal(sqrt.(m))) * Q' * M # sqrt by def (33), where M_ii = m_i^2
        q_tilde = Matrix(Diagonal(sqrt.(m))) * Q' * q
        U, s, V = LinearAlgebra.svd(M_tilde * inv_T) # transformation step 4
        d = U' * q_tilde # transformation step 5

        # unfolding steps 2 and 3 [hoecker1995svd]
        τ = s[n_df]^2 # deconvolution step 2
        z_τ = d .* s ./ ( s.^2 .+ τ )
        p_est = max.(0, inv_T * V * z_τ) # step 3 (denoted as w_tau in the paper)
        return p_est ./ sum(p_est)
    end

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    T = LinearAlgebra.diagm( # the Tikhonov matrix for optional curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]

    # least squares ||q - M*p||_2^2
    if strategy in [:softmax, :softmax_reg]
        @variable(model, l[1:(C-1)]) # latent variables (unconstrained), where l[C] = 0
        p = Vector{NonlinearExpression}(undef, C) # p = softmax(l)-1)
        for i in 1:(C-1)
            p[i] = @NLexpression(model, exp(l[i]) / (1 + sum(exp(l[j]) for j in 1:(C-1))))
        end
        p[C] = @NLexpression(model, 1 / (1 + sum(exp(l[j]) for j in 1:(C-1)))) # exp(0) = 1 for l[C] = 0
        if length(a) > 0
            @NLexpression(model, p_reg[i = 1:C], log10(1 + a[i] * p[i] * (N-C) / sum(a[j]*p[j] for j in 1:C)))
        else
            @NLexpression(model, p_reg[i = 1:C], p[i]) # just regularize 1/2*(Tp)^2
        end
        @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p_reg[j] for j in 1:C))
        if strategy == :softmax
            @NLobjective(model, Min,
                sum(((q[i] - sum(M[i, j] * p[j] for j in 1:C)) / w[i])^2 for i in 1:F) # loss function
                + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
            )
        else
            @NLobjective(model, Min,
                sum(((q[i] - sum(M[i, j] * p[j] for j in 1:C)) / w[i])^2 for i in 1:F)
                + τ/2 * sum(Tp[i]^2 for i in 1:(C-2))
                + λ * sum(l[j]^2 for j in 1:(C-1)) # soft-max regularization
            )
        end
    elseif strategy == :softmax_full_reg
        @variable(model, l[1:C]) # latent variables (unconstrained) with regularization
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
    _check_termination_status(termination_status(model), :least_squares, strategy, M, q)
    if strategy in [:softmax, :softmax_reg]
        exp_l = vcat(exp.(value.(l)), 1)
        return exp_l ./ sum(exp_l)
    elseif strategy == :softmax_full_reg
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy == :constrained
        return value.(p)
    end
end


function solve_maximum_likelihood(M::Matrix{Float64}, q::Vector{Float64}, N::Int, b::Vector{Float64}=zeros(length(q)); τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy::Symbol=:softmax, n_df::Int=size(M, 2), λ::Float64=1e-6)
    _check_solver_args(M, q)
    if any(sum(M; dims=2) .== 0) # limit the estimation to non-zero features
        nonzero = sum(M; dims=2)[:] .> 0
        q = q[nonzero]
        M = M[nonzero, :]
        b = b[nonzero]
    end
    F, C = size(M) # the numbers of features and classes
    T = LinearAlgebra.diagm( # the Tikhonov matrix for curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]

    if strategy == :original # here, we assume p contains counts, not probabilities
        q = 1 .+ (N-C) .* q # transform q to \bar{q}
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
            try
                diff = ForwardDiff.hessian!(diff, p -> sum(M*p - q .* log.(1 .+ M*p)), p_est)
            catch any_error
                if isa(any_error, DomainError)
                    @error "INVALID_MODEL with negative components in 1 .+ M*p"
                    throw(NonOptimalStatusError(INVALID_MODEL))
                else
                    rethrow()
                end
            end
            loss_p = DiffResults.value(diff)
            gradient_p = DiffResults.gradient(diff)
            hessian_p = DiffResults.hessian(diff)

            # check for convergence
            if abs(previous_loss - loss_p) < 1e-6
                break
            end
            previous_loss = loss_p

            # eigendecomposition of the Hessian: hessian_p == U*D*U'
            u, U = LinearAlgebra.eigen(hessian_p)
            if any(isa.(u, Complex)) || any(u .< 0) # occurs extremely rarely
                @warn "Assuming convergence from eigen-values that are not positive and real" u
                p_est = max.(0, p_est) # map to probabilities
                return p_est ./ sum(p_est)
            end
            D = Matrix(Diagonal(u .^ (-1/2))) # D^(-1/2)

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
    q = N .* q # transform q to \bar{q}
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    if strategy in [:softmax, :softmax_reg]
        @variable(model, l[1:(C-1)]) # latent variables (unconstrained), where l[C] = 0
        @NLexpression(model, exp_l[i = 1:(C-1)], exp(l[i]))
        @NLexpression(model, sum_exp_l, 1 + sum(exp_l[j] for j in 1:(C-1)))
        p = Vector{NonlinearExpression}(undef, C) # p = softmax(l)
        for i in 1:(C-1)
            p[i] = @NLexpression(model, exp_l[i] / sum_exp_l)
        end
        p[C] = @NLexpression(model, 1 / sum_exp_l) # exp(0) = 1 for l[C] = 0
        if strategy == :softmax
            @NLexpression(model, softmax_regularizer, 0.0) # no soft-max regularization
        else
            @NLexpression(model, softmax_regularizer, λ * sum(l[j]^2 for j in 1:(C-1)))
        end
    elseif strategy == :softmax_full_reg
        @variable(model, l[1:C]) # latent variables (unconstrained) with regularization
        @NLexpression(model, p[i = 1:C], exp(l[i]) / sum(exp(l[j]) for j in 1:C)) # p = softmax(l)
        @NLexpression(model, softmax_regularizer, λ * sum(l[j]^2 for j in 1:C))
    elseif strategy == :constrained
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @NLconstraint(model, sum(p[i] for i in 1:C) == 1)
        @NLexpression(model, softmax_regularizer, 0.0) # do not use soft-max regularization
    elseif strategy == :positive # ignore the sum constrained from strategy :constrained
        @variable(model, p[1:C] ≥ 0) # p_i ≥ 0
        @NLexpression(model, softmax_regularizer, 0.0)
    elseif strategy == :unconstrained
        @variable(model, p[1:C])
        @NLexpression(model, softmax_regularizer, 0.0)
    else
        error("There is no strategy \"$(strategy)\"")
    end

    # minimum regularized log-likelihood  ∑_{i=1}^F [Mp]_i - q_i ln [Mp]_i  +  τ/2 * (Tp)^2
    @NLexpression(model, Mp[i = 1:F], sum(M[i, j] * N * p[j] for j in 1:C))
    if length(a) > 0
        @NLexpression(model, p_reg[i = 1:C], log10(1 + a[i] * p[i] * (N-C) / sum(a[j]*p[j] for j in 1:C)))
        @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p_reg[j] for j in 1:C))
    else # just regularize 1/2*(Tp)^2
        @NLexpression(model, Tp[i = 1:(C-2)], sum(T[i, j] * p[j] for j in 1:C))
    end
    @NLobjective(model, Min,
        sum(Mp[i] - q[i] * log(Mp[i]) for i in 1:F) # loss function
        + τ/2 * sum(Tp[i]^2 for i in 1:(C-2)) # Tikhonov regularization
        + softmax_regularizer # optional soft-max regularization
    )

    # solve and return
    optimize!(model)
    _check_termination_status(termination_status(model), :maximum_likelihood, strategy, M, q)
    if strategy in [:softmax, :softmax_reg]
        exp_l = vcat(exp.(value.(l)), 1)
        return exp_l ./ sum(exp_l)
    elseif strategy == :softmax_full_reg
        return exp.(value.(l)) ./ sum(exp.(value.(l)))
    elseif strategy ∈ [:constrained, :unconstrained]
        return value.(p)
    elseif strategy == :positive
        return value.(p) ./ sum(value.(p))
    end
end

# brute-force search of a τ satisfying the n_df relation
function _select_τ(n_df::Number, eigvals_T::Vector{Float64}, min::Float64=-12.0, max::Float64=6.0, i::Int64=2)
    τ_candidates = 10 .^ range(min, stop=max, length=100)
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


function solve_hellinger_distance(M::Matrix{Float64}, q::Vector{Float64}, N::Int, n_bins::Int, b::Vector{Float64}=zeros(length(q)); τ::Float64=0.0, a::Vector{Float64}=Float64[], strategy::Symbol=:softmax, λ::Float64=1e-6, n_trials=10)
    _check_solver_args(M, q)
    indices = [ (1+(i-1)*n_bins):(i*n_bins) for i in 1:Int(size(M, 1) / n_bins) ]
    if any(sum(M; dims=2) .== 0) # limit the estimation to non-zero features
        nonzero = sum(M; dims=2)[:] .> 0
        q = q[nonzero]
        M = M[nonzero, :]
        b = b[nonzero]
        i = 1
        indices = map(indices) do indices_in
            indices_out = Int[]
            for j in indices_in
                if nonzero[j]
                    push!(indices_out, i)
                    i += 1
                end
            end
            indices_out
        end
    end
    M = M .* (sum(q)^2 / (sum(b) + sum(q))) ./ sum(M; dims=1)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    F, C = size(M) # the numbers of "multi-features" and classes
    n_features = length(indices) # the number of actual features in X
    T = LinearAlgebra.diagm( # the Tikhonov matrix for curvature regularization
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]

    # set up the solution vector p
    if strategy in [:softmax, :softmax_reg]
        @variable(model, l[1:(C-1)]) # latent variables (unconstrained), where l[C] = 0
        p = Vector{NonlinearExpression}(undef, C) # p = softmax(l)
        for i in 1:(C-1)
            p[i] = @NLexpression(model, exp(l[i]) / (1 + sum(exp(l[j]) for j in 1:(C-1))))
        end
        p[C] = @NLexpression(model, 1 / (1 + sum(exp(l[j]) for j in 1:(C-1)))) # exp(0) = 1 for l[C] = 0
        if strategy == :softmax
            @NLexpression(model, softmax_regularizer, 0.0) # no soft-max regularization
        else
            @NLexpression(model, softmax_regularizer, λ * sum(l[j]^2 for j in 1:(C-1)))
        end
    elseif strategy == :softmax_full_reg
        @variable(model, l[1:C]) # latent variables (unconstrained) with regularization
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
    # @NLexpression(model, ap[i = 1:C], a[i] * p[i] * N / sum(a[j]*p[j] for j in 1:C))
    # @NLexpression(model, Mp[i = 1:F], (sum(M[i, j] * ap[j] for j in 1:C) + α * b[i]) / (N + α * N_b))
    @NLexpression(model, Mp[i = 1:F], sum(M[i, j] * p[j] for j in 1:C) + b[i])
    @NLexpression(model, squared[i = 1:F], (sqrt(q[i]) - sqrt(Mp[i]))^2)
    @NLexpression(model, HD[i = 1:n_features], sqrt(sum((squared[j] for j in indices[i]))))
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

    # solve: HD requires multiple trials because derivatives are not globally defined
    best_p = (Inf, Float64[])
    first_p = Inf # just for logging the improvement
    for i_trial in 1:n_trials
        if strategy in [:softmax, :softmax_reg]
            set_start_value.(l, rand(C-1) .* 2 .- 1)
        elseif strategy == :softmax_full_reg
            set_start_value.(l, rand(C) .* 2 .- 1)
        elseif strategy == :constrained
            p_0 = rand(C)
            set_start_value.(p, p_0 ./ sum(p_0))
        end
        optimize!(model)
        if termination_status(model) != INVALID_MODEL
            _check_termination_status(termination_status(model), :hellinger_distance, strategy, M, q)
        end # otherwise, continue with an INVALID_MODEL result
        if objective_value(model) < best_p[1]
            if strategy in [:softmax, :softmax_reg]
                exp_l = vcat(exp.(value.(l)), 1)
                best_p = (objective_value(model), exp_l ./ sum(exp_l))
            elseif strategy == :softmax_full_reg
                best_p = (objective_value(model), exp.(value.(l)) ./ sum(exp.(value.(l))))
            elseif strategy == :constrained
                best_p = (objective_value(model), value.(p))
            end
        end
        if i_trial == 1
            first_p = objective_value(model)
        end
    end
    @debug "hellinger_distance ($strategy)" best_p[1] first_p
    return best_p[2]
end

function solve_expectation_maximization(M::Matrix{Float64}, q::Vector{Float64}, N::Int, p_0::Vector{Float64}; o::Int=-1, λ::Float64=.0, a::Vector{Float64}=Float64[], n_iterations::Int=100, ϵ::Float64=.0)
    if any(sum(M; dims=2) .== 0) # limit the estimation to non-zero features
        nonzero = sum(M; dims=2)[:] .> 0
        q = q[nonzero]
        M = M[nonzero, :]
    end
    F, C = size(M) # the numbers of features and classes
    p_est = zeros(C) # the estimate
    p_prev = p_0 # unsmoothed previous estimate for convergence check
    for _ ∈ 1:n_iterations
        Mp = M .* p_0' # element-wise multiplication, [M]_ij * [p]_i
        for i ∈ 1:C
            p_est[i] = sum(Mp[j,i] * q[j] / sum(Mp[j,:]) for j ∈ 1:F)
        end
        p_0 = λ > 0 ? _smooth(p_est, N, o, λ, a) : p_est # prior of the next iteration
        if _chisquare_distance(p_est, p_prev) < ϵ
            break # assume convergence
        end
        p_prev = p_est
    end
    return p_est
end

_chisquare_distance(x::Vector{Float64}, y::Vector{Float64}) =
    sum((x - y).^2 / (x + y)) # https://github.com/JuliaStats/Distances.jl

function _smooth(p_est::Vector{Float64}, N::Int, o::Int, λ::Float64, a::Vector{Float64})
    C = length(p_est)
    p_o = if length(a) > 0 # fit and apply a polynomial to a log10 acceptance correction
            (10 .^ Polynomials.fit(
                    Float64.(1:C),
                    log10.(1 .+ p_est .* a .* (N-C) ./ sum(p_est .* a)),
                    o
                ).(1:C)) ./ a # also transform back to a non-log non-acceptance version
        else # fit and apply a polynomial to p_est
            max.(0, Polynomials.fit(Float64.(1:C), p_est, o).(1:C))
        end
    p_o ./= sum(p_o) # normalize to a probability
    p_o = λ * p_o + (1-λ) * p_est # linear interpolation
    return p_o ./= sum(p_o) # normalize again
end
