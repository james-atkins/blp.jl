export Market, compute_mu, compute_delta, I, J, K2

struct Market{T<:AbstractFloat}
    # TODO: ultimately a market is going to be a view of the larger dataset so these
    # owning types should be changed to parameterised types.
    x2::Matrix{T}      # J x K2
    shares::Vector{T}  # J
    weights::Vector{T} # I
    nodes::Matrix{T}   # I x K2

    # TODO: check correct dimensions
end

Base.eltype(::Type{Market{T}}) where {T} = T

""" Individuals in market. """
num_individuals(market::Market) = length(market.weights)

""" Products in market. """
num_products(market::Market) = size(market.x2, 1)

""" Demand-side nonlinear product characteristics. """
num_nonlinear_demand_characteristics(market::Market) = size(market.x2, 2)

""" "Compute the mean utility that solves the simple logit model. """
function compute_logit_delta(market)
    # log1p(-x) ≡ log(1-x) but faster and more accurately
    return log.(market.shares) .- log1p(-sum(market.shares))
end

""" Compute the mean utility for this market that equates observed and predicted market shares. """
function compute_delta(market::Market, iteration, sigma)
    result, _, _ = compute_delta_impl(market, iteration, sigma)
    return result
end

function compute_mu(market::Market, sigma)
    # Currently just support unobserved heterogeneity.
    return market.x2 * sigma * market.nodes'
end

# This code is used for both solving the contraction mapping as-is and for
# rules for automatic differentiation.
function compute_delta_impl(market::Market, iteration, sigma)
    mu = compute_mu(market, sigma)
    log_market_shares = log.(market.shares)

    I = num_individuals(market)
    J = num_products(market)

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

    return fixed_point_iteration(iteration, compute_logit_delta(market), contraction!), shares, probabilities
end
