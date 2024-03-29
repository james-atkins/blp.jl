function solve_mpec(problem::Problem, theta2::Theta2, knitro_options...)
    check_compatible_theta2(problem, theta2)

    N = problem.N
    X1 = [problem.products.X1_exog problem.products.prices]
    Z = problem.products.Z_demand
    W = Symmetric(inv(cholesky(Z' * Z)))

    # This is nothing MPEC specific and should be extracted for use by solve_nfp too
    theta2_flat, unflatten = flatten(theta2)

    @info "Running GMM step 1"
    mpec = MPECProblem(problem, unflatten, IVGMM(X1, Z, W))
    result = solve(mpec, W, knitro_options...)
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

    # Second stage
    @info "Running GMM step 2"
    g = gmm_moments(result)
    centered_g = g .- mean(g, dims = 1)

    # Compute heteroscedasticity robust weighting matrix
    W2 = Symmetric(inv(chol_tcrossprod(centered_g)))

    mpec2 = MPECProblem(problem, unflatten, IVGMM(X1, Z, W2))
    result = solve(mpec2, W2, knitro_options...)
    if !gmm_success(result)
        error("Knitro could not solve MPEC")
    end

    params = gmm_estimate(result)
    delta_solved = params[1:N]
    theta2_flat_solved = params[N+1:end]
    theta2_solved = unflatten(theta2_flat_solved)

    theta1_hat = estimate(mpec.ivgmm, delta_solved)

    shares, probabilities = compute_shares_and_choice_probabilities(problem, delta_solved, theta2_solved)

    P = div(problem.K2 * (problem.K2 + 1), 2) + (problem.K2 * problem.D)

    blocks = [Matrix{eltype(problem)}(undef, market.J, market.J) for market in problem.markets]
    jacobian_shares_by_delta = BlockDiagonal(blocks)
    jacobian_shares_by_theta2 = Matrix{eltype(problem)}(undef, N, P)

    jacobian_shares_by_delta!(problem, shares, probabilities, jacobian_shares_by_delta)
    jacobian_shares_by_theta2!(problem, probabilities, jacobian_shares_by_theta2)

    jacobian_delta_by_theta2 = -jacobian_shares_by_delta \ jacobian_shares_by_theta2
    G = Z' * jacobian_delta_by_theta2

    g = gmm_moments(result)
    centered_g = g .- mean(g, dims = 1)

    B = centered_g' * centered_g
    W2_chol = cholesky(W2)
    bread = Symmetric(inv(chol_tcrossprod(W2_chol.U * G)))

    V = bread * G' * W2 * B * W2 * G * bread

    # Delta rule
    grad_h = mpec.ivgmm.S * jacobian_delta_by_theta2
    V_theta1 = grad_h * V * grad_h'

    return MPECSolution(
        objective_value = gmm_objective_value(result),
        theta1 = theta1_hat,
        theta1_stderrs = sqrt.(diag(V_theta1)),
        theta2 = theta2_solved,
        theta2_stderrs = sqrt.(diag(V)),
        delta = delta_solved,
        probabilities = probabilities,
        shares = shares,
    )
end

Base.@kwdef struct MPECSolution
    objective_value::Float64
    theta1::Vector{Float64}
    theta1_stderrs::Vector{Float64}
    theta2::Theta2
    theta2_stderrs::Vector{Float64}
    delta::Vector{Float64}
    probabilities::Vector{Matrix{Float64}}
    shares::Vector{Float64}
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
