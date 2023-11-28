function solve_mpec(problem::Problem, theta2::Theta2)
    check_compatible_theta2(problem, theta2)

    N = problem.N
    X1 = [problem.products.X1_exog problem.products.prices]
    Z = problem.products.Z_demand
    W = Symmetric(inv(cholesky(Z' * Z)))

    # This is nothing MPEC specific and should be extracted for use by solve_nfp too
    theta2_flat, unflatten = flatten(theta2)

    mpec = MPECProblem(problem, unflatten, IVGMM(X1, Z, W))
    result = solve(mpec, W)
    if !gmm_success(result)
        error("Knitro could not solve MPEC")
    end

    params = gmm_estimate(result)

    # Need to compute initial delta
    # Gaston suggested that we solve the contraction mapping for our guess of theta2,
    # and then use that to kickstart the MPEC.
    # @info "Solving for inital delta"
    # initial_delta = solve_demand(problem, theta2, SQUAREMIteration())
    # @info "Solved for inital delta"

    # initial_params = [ initial_delta ; theta2_flat ]
    # _, params = solve(mpec, W, initial_theta = initial_params)

    delta_solved = params[1:N]
    theta2_flat_solved = params[N+1:end]
    theta2_solved = unflatten(theta2_flat_solved)

    # Retrieve theta 1
    theta1_hat = estimate(mpec.ivgmm, delta_solved)
    @info theta1_hat

    # Second stage
    @info "Running second stage"
    g = gmm_moments(result)
    centered_g = g .- mean(g, dims = 1)

    # Compute heteroscedasticity robust weighting matrix
    W2 = Symmetric(inv(chol_tcrossprod(centered_g)))

    mpec2 = MPECProblem(problem, unflatten, IVGMM(X1, Z, W2))
    result = solve(mpec2, W2)
    if !gmm_success(result)
        error("Knitro could not solve MPEC")
    end

    params = gmm_estimate(result)
    delta_solved = params[1:N]
    theta2_flat_solved = params[N+1:end]
    theta2_solved = unflatten(theta2_flat_solved)

    theta1_hat = estimate(mpec.ivgmm, delta_solved)
    @info theta1_hat

    return theta2_solved
end

struct MPECProblem{T <: AbstractFloat} <: GMMModel
    p::Problem{T}
    unflatten::Function
    ivgmm::IVGMM{T}
end

function gmm_num_residuals(problem::MPECProblem)
    return Int64(problem.p.N)
end

function gmm_num_instruments(problem::MPECProblem)
    return size(problem.p.products.Z_demand, 2)
    return Int64(problem.p.M_demand)
end

function gmm_num_constraints(problem::MPECProblem)
    return Int64(problem.p.N)
end

function gmm_num_parameters(problem::MPECProblem)
    # δ + θ₂
    K2 = problem.p.K2
    D = problem.p.D
    P = div(K2 * (K2 + 1), 2) + (K2 * D)
    return Int64(problem.p.N + P)
end

function gmm_instruments(problem::MPECProblem)
    return problem.p.products.Z_demand
end

function gmm_residuals_constraints!(problem::MPECProblem, params, residuals, constraints)
    T = eltype(problem.p)
    N = problem.p.N
    K2 = problem.p.K2
    D = problem.p.D
    P = div(K2 * (K2 + 1), 2) + (K2 * D)

    @assert T != Any

    delta, theta2_flat = params[1:N], params[N+1:end]
    @assert length(delta) == N
    @assert length(theta2_flat) == P

    theta2 = problem.unflatten(theta2_flat)

    # TODO: These should be preallocated in advance and reused across loops
    probabilities = [Matrix{T}(undef, market.J, market.I) for market in problem.p.markets]

    ### Compute the residuals r(δ, θ₂) = ξ(δ, θ₂)
    if residuals !== nothing
        @assert length(residuals) == N
        residuals!(problem.ivgmm, delta, residuals)
    end

    ### Compute the constraint c(δ, θ₂) = s(δ, θ₂) - S
    if constraints !== nothing
        @assert length(constraints) == N
        compute_shares_and_choice_probabilities!(problem.p, delta, theta2, constraints, probabilities)
        @. constraints -= problem.p.products.shares
    end
end

function gmm_residuals_constraints_jacobians!(problem::MPECProblem, params, residuals_jacobian, constraints_jacobian)
    T = eltype(problem.p)
    N = problem.p.N
    K2 = problem.p.K2
    D = problem.p.D
    P = div(K2 * (K2 + 1), 2) + (K2 * D)

    @assert T != Any
    @assert size(residuals_jacobian) == (N, N + P)
    @assert size(constraints_jacobian) == (N, N + P)

    delta, theta2_flat = params[1:N], params[N+1:end]
    @assert length(delta) == N
    @assert length(theta2_flat) == P
    theta2 = problem.unflatten(theta2_flat)

    # TODO: These should be preallocated in advance and reused across loops
    shares = Vector{T}(undef, N)
    probabilities = [Matrix{T}(undef, market.J, market.I) for market in problem.p.markets]
    blocks = [Matrix{T}(undef, market.J, market.J) for market in problem.p.markets]
    jacobian_shares_by_delta = BlockDiagonal(blocks)
    jacobian_shares_by_theta2 = Matrix{T}(undef, N, P)

    ## Jacobian of the residuals
    # ∂r/∂δ = ∂ξ/∂δ  = X(X'ZWZ'X)^-1 X'ZWZ'  (NxN)
    residuals_jacobian!(problem.ivgmm, @view residuals_jacobian[:, 1:N])
    residuals_jacobian[:, N+1:end] .= 0

    ## The Jacobian of the constraints
    # ∂c/∂δ = ∂s/∂δ  (NxN)
    # ∂c/∂θ₂= ∂s/∂θ  (NxP)
    compute_shares_and_choice_probabilities!(problem.p, delta, theta2, shares, probabilities)
    jacobian_shares_by_delta!(problem.p, shares, probabilities, jacobian_shares_by_delta)
    jacobian_shares_by_theta2!(problem.p, probabilities, jacobian_shares_by_theta2)

    constraints_jacobian[:, 1:N] = jacobian_shares_by_delta
    constraints_jacobian[:, N+1:end] = jacobian_shares_by_theta2
end


function solve_mpec_supply(problem::Problem, theta2::Theta2)
    check_compatible_theta2(problem, theta2)

    N = problem.N
    Z = problem.products.Z_demand
    W = Symmetric(inv(chol_tcrossprod(Z)))

    # This is nothing MPEC specific and should be extracted for use by solve_nfp too
    theta2_flat, unflatten = flatten(theta2)

    mpec = MPECSupply(problem, unflatten, IVGMM(problem.products.X1_exog, Z, W))
    result = solve(mpec, W)
    if !gmm_success(result)
        error("Knitro could not solve MPEC")
    end

    params = gmm_estimate(result)

    delta_solved = params[1:N]
    theta2_flat_solved = params[N+1:end]
    theta2_solved = unflatten(theta2_flat_solved)

    @info theta2_solved

    # Retrieve theta 1
    theta1_hat = estimate(mpec.ivgmm, delta_solved .- (theta2_solved.alpha .* problem.products.prices))
    @info theta1_hat
end

struct MPECSupply{T <: AbstractFloat} <: GMMModel
    p::Problem{T}
    unflatten::Function
    ivgmm::IVGMM{T}
end

gmm_num_residuals(mpec::MPECSupply) = Int64(mpec.p.N)
gmm_num_instruments(mpec::MPECSupply) = Int64(mpec.p.M_demand)
gmm_num_constraints(mpec::MPECSupply) = Int64(mpec.p.N)
gmm_instruments(mpec::MPECSupply) = mpec.p.products.Z_demand

function gmm_num_parameters(mpec::MPECSupply)
    # δ + θ₂
    K2 = mpec.p.K2
    D = mpec.p.D
    P = div(K2 * (K2 + 1), 2) + (K2 * D) + 1  # Includes alpha
    return Int64(mpec.p.N + P)
end

function gmm_residuals_constraints!(problem::MPECSupply, params, residuals, constraints)
    T = eltype(problem.p)
    N = problem.p.N
    K2 = problem.p.K2
    D = problem.p.D
    P = div(K2 * (K2 + 1), 2) + (K2 * D) + 1  # Includes alpha

    @assert T != Any

    delta, theta2_flat = params[1:N], params[N+1:end]
    @assert length(delta) == N
    @assert length(theta2_flat) == P

    theta2 = problem.unflatten(theta2_flat)

    # TODO: These should be preallocated in advance and reused across loops
    probabilities = [Matrix{T}(undef, market.J, market.I) for market in problem.p.markets]

    ### Compute the residuals r(δ-αp, θ₂) = ξ(δ-αp, θ₂)
    if residuals !== nothing
        @assert length(residuals) == N
        residuals!(problem.ivgmm, delta .- (theta2.alpha .* problem.p.products.prices), residuals)
    end

    ### Compute the constraint c(δ, θ₂) = s(δ, θ₂) - S
    if constraints !== nothing
        @assert length(constraints) == N
        compute_shares_and_choice_probabilities!(problem.p, delta, theta2, constraints, probabilities)
        @. constraints -= problem.p.products.shares
    end
end

function gmm_residuals_constraints_jacobians!(problem::MPECSupply, params, residuals_jacobian, constraints_jacobian)
    T = eltype(problem.p)
    N = problem.p.N
    K2 = problem.p.K2
    D = problem.p.D
    P = div(K2 * (K2 + 1), 2) + (K2 * D) + 1  # Includes alpha

    @assert T != Any
    @assert size(residuals_jacobian) == (N, N + P)
    @assert size(constraints_jacobian) == (N, N + P)

    delta, theta2_flat = params[1:N], params[N+1:end]
    @assert length(delta) == N
    @assert length(theta2_flat) == P
    theta2 = problem.unflatten(theta2_flat)

    # TODO: These should be preallocated in advance and reused across loops
    shares = Vector{T}(undef, N)
    probabilities = [Matrix{T}(undef, market.J, market.I) for market in problem.p.markets]
    blocks = [Matrix{T}(undef, market.J, market.J) for market in problem.p.markets]
    jacobian_shares_by_delta = BlockDiagonal(blocks)
    jacobian_shares_by_theta2 = Matrix{T}(undef, N, P-1)

    ## Jacobian of the residuals
    # ∂r/∂δ = ∂ξ/∂δ  = I - X(X'ZWZ'X)^-1 X'ZWZ'      (NxN)
    # ∂r/∂θ₂= ∂ξ/∂θ₂ = 0                             (NxP-1)
    # ∂r/∂α = ∂ξ/∂α  = -[I - X(X'ZWZ'X)^-1 X'ZWZ']p  (Nx1)
    res_jac_delta = @view residuals_jacobian[:, 1:N]
    res_jac_theta2 = @view residuals_jacobian[:, N+1:N+P-1]
    res_jac_alpha = @view residuals_jacobian[:, N+P]

    residuals_jacobian!(problem.ivgmm, res_jac_delta)
    res_jac_theta2 .= 0
    mul!(res_jac_alpha, res_jac_delta, problem.p.products.prices, -1, 0)

    ## The Jacobian of the constraints
    # ∂c/∂δ = ∂s/∂δ  (NxN)
    # ∂c/∂θ₂= ∂s/∂θ  (NxP-1)
    # ∂c/∂α = 0      (Nx1)
    compute_shares_and_choice_probabilities!(problem.p, delta, theta2, shares, probabilities)
    jacobian_shares_by_delta!(problem.p, shares, probabilities, jacobian_shares_by_delta)
    jacobian_shares_by_theta2!(problem.p, probabilities, jacobian_shares_by_theta2)

    constraints_jacobian[:, 1:N] = jacobian_shares_by_delta
    constraints_jacobian[:, N+1:N+P-1] = jacobian_shares_by_theta2
    constraints_jacobian[:, N+P] .= 0
end
