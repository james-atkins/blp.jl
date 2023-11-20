export Market, NLSolveInversion, compute_mu, solve_demand

struct Market{
    T <: AbstractFloat,
    Shares <: AbstractVector{T},
    X2 <: AbstractMatrix{T},
    Weights <: AbstractVector{T},
    Tastes <: AbstractMatrix{T},
    Demographics <: AbstractMatrix{T},
}
    shares::Shares              # J
    x2::X2                      # J x K2
    weights::Weights            # I
    tastes::Tastes              # I x K2
    demographics::Demographics  # I x D

    J::Int32   # Products in market
    I::Int32   # Individuals in market
    K2::Int32  # Number of demand-side nonlinear product characteristics
    D::Int32   # Number of demographic variables

    function Market(shares, x2, weights, tastes, demographics)
        if !allequal([size(x, 1) for x in [shares, x2]])
            throw(DimensionMismatch())
        end

        if !allequal([size(x, 1) for x in [weights, tastes, demographics]])
            throw(DimensionMismatch())
        end

        if !allequal([size(x, 2) for x in [x2, tastes]])
            throw(DimensionMismatch())
        end

        J, K2 = size(x2)
        I, D = size(demographics)

        T = promote_type(eltype(shares), eltype(x2), eltype(weights), eltype(demographics))
        new{T, typeof(shares), typeof(x2), typeof(weights), typeof(tastes), typeof(demographics)}(
            shares,
            x2,
            weights,
            tastes,
            demographics,
            J,
            I,
            K2,
            D,
        )
    end
end

function Market(shares, x2, weights, tastes)
    demographics = Matrix{Float64}(undef, length(weights), 0)
    return Market(shares, x2, weights, tastes, demographics)
end

Base.eltype(
    ::Type{Market{T, Shares, X2, Weights, Tastes, Demographics}},
) where {T, Shares, X2, Weights, Tastes, Demographics} = T

""" "Compute the mean utility that solves the simple logit model. """
function logit_delta(market)
    # log1p(-x) ≡ log(1-x) but faster and more accurately
    return log.(market.shares) .- log1p(-sum(market.shares))
end

function compute_mu(market::Market, theta2::Theta2)
    # Returns a J x I matrix
    return market.x2 * ((theta2.sigma * market.tastes') + (theta2.pi * market.demographics'))
end

function compute_shares_and_choice_probabilities!(
    market::Market,
    delta::AbstractVector,
    theta2::Theta2,
    shares::AbstractVector,
    probabilities::AbstractMatrix,
)
    utilities = delta .+ compute_mu(market, theta2)

    choice_probabilities!(probabilities, utilities)
    mul!(shares, probabilities, market.weights)
end


Base.@kwdef struct NLSolveInversion
    method = :newton
    iterations = 1_000
    ftol::Float64 = 0.0
end


"""
Solve for the mean utility for this market that equates observed and predicted market shares.

Return a tuple (delta, shares, probabilities).

This method uses the NLSolve package.
"""
function solve_demand(market::Market, theta2::Theta2, config::NLSolveInversion; tolerance = 1E-14)
    mu = compute_mu(market, theta2)

    # Speed is imperative for solving the inner loop so avoid unnecessary memory allocations by
    # pre-allocating and then using in-place operations. Unfortunately, this makes the code a bit
    # harder to understand.
    utilities = similar(mu)
    probs = similar(mu)
    weighted_probs = similar(mu)
    utilities = similar(mu)
    shares = similar(market.shares)

    function fj!(F, J, delta)
        @. utilities = delta + mu
        choice_probabilities!(probs, utilities)

        # Integrate over individuals to compute market shares
        mul!(shares, probs, market.weights)

        # Compute difference between predicted and observed market shares
        if F !== nothing
            @. F = shares - market.shares
        end

        # Compute Jacobian of predicted market shares with respect to delta
        if J !== nothing
            @. weighted_probs = probs * market.weights'

            # In-place equivalent to Diagonal(shares) - (probs * weighted_probs')
            J[:] = Diagonal(shares)
            mul!(J, probs, weighted_probs', -1, 1)
        end
    end

    initial_delta = logit_delta(market)

    local res
    try
        res = nlsolve(
            only_fj!(fj!),
            initial_delta;
            xtol = tolerance,
            iterations = config.iterations,
            ftol = config.ftol,
            method = config.method,
        )
    catch e
        if e isa IsFiniteException
            return InversionResult(INVERSION_NUMERICAL_ISSUES, initial_delta, 0, 0), shares, probs
        else
            rethrow(e)
        end
    end

    if !converged(res)
        return InversionResult(INVERSION_EXCEEDED_MAX_ITERATIONS, res.zero, res.iterations, res.f_calls), shares, probs
    end

    return InversionResult(INVERSION_CONVERGED, res.zero, res.iterations, res.f_calls), shares, probs
end

"""
Solve for the mean utility for this market that equates observed and predicted market shares.

Return a tuple (delta, shares, probabilities).

This method uses the BLP contraction mapping approach.
"""
function solve_demand(market::Market, theta::Theta2, iteration::Iteration; tolerance = 1E-14)
    mu = compute_mu(market, theta)
    log_market_shares = log.(market.shares)

    # Speed is imperative for the contraction mapping so avoid unnecessary memory allocations by
    # pre-allocating and then using in-place operations. Unfortunately, this makes the code a bit
    # harder to understand.
    utilities = similar(mu)
    probabilities = similar(mu)
    shares = similar(market.shares)

    function contraction!(delta_out, delta_in)
        @. utilities = delta_in + mu
        choice_probabilities!(probabilities, utilities)
        mul!(shares, probabilities, market.weights)  # Compute market share by integrating over individuals
        @. delta_out = delta_in + log_market_shares - log(shares)
    end

    return fixed_point_iteration(iteration, logit_delta(market), contraction!, tolerance = tolerance), shares, probabilities
end


function jacobian_shares_by_delta!(market::Market, shares::AbstractVector, probabilities::AbstractMatrix, jacobian::AbstractMatrix)
    weighted_probabilities = probabilities .* market.weights'

    # In-place equivalent to Diagonal(shares) - (probs * weighted_probs')
    jacobian[:] = Diagonal(shares)
    mul!(jacobian, probabilities, weighted_probabilities', -1, 1)
end

function jacobian_shares_by_delta(market::Market, probabilities::AbstractMatrix)
    shares = probabilities * market.weights
    jacobian = Matrix{eltype(market)}(undef, market.J, market.J)

    jacobian_shares_by_delta!(market, shares, probabilities, jacobian)
    return jacobian
end


function jacobian_shares_by_theta2!(market::Market, probabilities::AbstractMatrix, jacobian::AbstractMatrix)
    p = 1

    # Loop over the indices of sigma, i.e. indices of lower triangle matrix
    for col_idx = 1:market.K2
        for row_idx = col_idx:market.K2
            x = @view market.x2[:, row_idx]      # Product characteristics
            v = @view market.tastes[:, col_idx]  # Individual characteristics
            jacobian[:, p] = (probabilities .* v' .* (x .- x' * probabilities)) * market.weights
            p += 1
        end
    end

    # Loop over the indices of pi
    for col_idx = 1:market.D
        for row_idx = 1:market.K2
            x = @view market.x2[:, row_idx]            # Product characteristics
            v = @view market.demographics[:, col_idx]  # Individual characteristics
            jacobian[:, p] = (probabilities .* v' .* (x .- x' * probabilities)) * market.weights
            p += 1
        end
    end
end

function jacobian_shares_by_theta2(market::Market, probabilities::AbstractMatrix)
    P = div(market.K2 * (market.K2 + 1), 2) + (market.K2 * market.D)
    jacobian = Matrix{eltype(market)}(undef, market.J, P)

    jacobian_shares_by_theta2!(market, probabilities, jacobian)
    return jacobian
end
