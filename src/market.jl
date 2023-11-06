export Market, Theta2, NLSolveInversion, compute_mu, compute_delta

struct Theta2
	sigma  # K2 x K2 lower triangle
	pi     # K2 x D
end

function Theta2(sigma)
    K2 = size(sigma, 1)
    pi = Matrix{eltype(sigma)}(undef, K2, 0)
    return Theta2(sigma, pi)
end

struct Market{T<:AbstractFloat}
    # TODO: ultimately a market is going to be a view of the larger dataset so these
    # owning types should be changed to parameterised types.
    shares::Vector{T}        # J
	X2::Matrix{T}            # J x K2
	weights::Vector{T}       # I
	tastes::Matrix{T}        # I x K2
	demographics::Matrix{T}  # I x D

	J::Int32  # Products in market
	I::Int32  # Individuals in market
	K2::Int32 # Number of demand-side nonlinear product characteristics
	D::Int32  # Number of demographic variables

    function Market(shares, X2, weights, tastes, demographics)
        if !allequal([size(x, 1) for x in [shares, X2]])
            throw(DimensionMismatch())
        end

        if !allequal([size(x, 1) for x in [weights, tastes, demographics]])
            throw(DimensionMismatch())
        end

        if !allequal([size(x, 2) for x in [X2, tastes]])
            @info size(X2)
            @info size(tastes)
            throw(DimensionMismatch())
        end

        J, K2 = size(X2)
        I, D = size(demographics)

        T = Union{eltype(shares), eltype(X2), eltype(weights), eltype(demographics)}

        new{T}(shares, X2, weights, tastes, demographics, J, I, K2, D)
    end
end

function Market(shares, X2, weights, tastes)
    demographics = Matrix{Float64}(undef, length(weights), 0)
    return Market(shares, X2, weights, tastes, demographics)
end

Base.eltype(::Type{Market{T}}) where {T} = T

""" "Compute the mean utility that solves the simple logit model. """
function logit_delta(market)
    # log1p(-x) ≡ log(1-x) but faster and more accurately
    return log.(market.shares) .- log1p(-sum(market.shares))
end

function compute_mu(market::Market, theta2::Theta2)
    # Returns a J x I matrix
	return market.X2 * ((theta2.sigma * market.tastes') + (theta2.pi * market.demographics'))
end

struct NLSolveInversion end

"""
Solve for the mean utility for this market that equates observed and predicted market shares.
"""
function compute_delta(market::Market, theta2::Theta2, ::NLSolveInversion)
	mu = compute_mu(market, theta2)

	# Speed is imperative for solving the inner loop so avoid unnecessary memory allocations by
    # pre-allocating and then using in-place operations. Unfortunately, this makes the code a bit
    # harder to understand.
    utilities = similar(mu)
    probs = similar(mu)
    weighted_probs = similar(mu)
    utilities = similar(mu)
    shares = Vector{eltype(market)}(undef, market.J)

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
	return nlsolve(only_fj!(fj!), initial_delta)
end

"""
Solve for the mean utility for this market that equates observed and predicted market shares.

This method uses the BLP contraction mapping approach.
"""
function compute_delta(market::Market, theta::Theta2, iteration::Iteration)
    mu = compute_mu(market, theta)
    log_market_shares = log.(market.shares)

    J, I = market.J, market.I

    # Speed is imperative for the contraction mapping so avoid unnecessary memory allocations by
    # pre-allocating and then using in-place operations. Unfortunately, this makes the code a bit
    # harder to understand.
    utilities = Matrix{eltype(market)}(undef, J, I)
    probabilities = Matrix{eltype(market)}(undef, J, I)
    shares = Vector{eltype(market)}(undef, J)

    function contraction!(delta_out::AbstractVector, delta_in::AbstractVector)
        @. utilities = delta_in + mu
        choice_probabilities!(probabilities, utilities)
        mul!(shares, probabilities, market.weights)  # Compute market share by integrating over individuals
        @. delta_out = delta_in + log_market_shares - log(shares)
    end

    return fixed_point_iteration(iteration, logit_delta(market), contraction!)
end
