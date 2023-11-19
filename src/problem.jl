Base.@kwdef struct Problem{T <: AbstractFloat}
    # Products
    shares::Vector{T}    # J
    X1_exog::Matrix{T}   # J x K1_exog
    X1_endog::Matrix{T}  # J x K1_endo
    X2::Matrix{T}        # J x K2
    Z_demand::Matrix{T}  # J x M_demand

    # Individuals
    weights::Vector{T}       # I
    tastes::Matrix{T}        # I x K2
    demographics::Matrix{T}  # I x D
    
    markets::Vector{Market}
    markets_masks::Vector{UnitRange{Int64}}

    N::Int32         # Number of products
    K1::Int32        # Number of demand-side linear product characteristics
    K2::Int32        # Number of demand-side nonlinear product characteristics
    M_demand::Int32  # Number of demand-side instruments
    I::Int32         # Number of individuals
    D::Int32         # Number of demographic variables
end

Base.eltype(::Type{Problem{T}}) where {T} = T

"""
Given guesses of `delta` and `theta2`, compute the choice probabilities and market shares.
"""
function compute_shares_and_choice_probabilities!(problem::Problem, delta::AbstractVector, theta2::Theta2, shares::AbstractVector, probabilities::AbstractMatrix)
    if length(shares) != problem.N
        throw(DimensionMismatch("shares has invalid length. expected $(problem.N); actual $(length(shares))"))
    end

    if size(probabilities, 1) != problem.N
        throw(DimensionMismatch("probabilities has invalid size. expected: $(problem.N); actual: $(size(probabilities, 1))"))
    end

    Threads.@threads for (market, mask) in zip(problem.markets, problem.markets_masks)
        delta_market = @view delta[mask]
        shares_market = @view shares[mask]
        probabilities_market = @view probabilities[mask, :]

        compute_shares_and_choice_probabilities!(market, delta_market, theta2, shares_market, probabilities_market)
    end
end

function jacobian_shares_by_delta!(problem::Problem, shares::AbstractVector, probabilities::AbstractMatrix, jacobian::BlockDiagonal)
    if length(problem.markets) != length(jacobian.blocks)
        throw(DimensionMismatch("jacobian has invalid number of blocks. expected $(length(problem.markets)); actual $(length(jacobian.blocks))"))
    end

    Threads.@threads for (market, mask, block) in zip(problem.markets, problem.markets_masks, jacobian.blocks)
        shares_market = @view shares[mask]
        probabilities_market = @view probabilities[mask, :]

        jacobian_shares_by_delta!(market, shares_market, probabilities_market, block)
    end
end


function jacobian_shares_by_theta2!(problem::Problem, probabilities::AbstractMatrix, jacobian::AbstractMatrix)
    P = div(problem.K2 * (problem.K2 + 1), 2) + (problem.K2 * problem.D)
    if size(jacobian) != (problem.N, P)
        throw(DimensionMismatch("jacobian has invalid size. expected $((problem.N, P)); actual $(size(probabilities))"))
    end

    Threads.@threads for (market, mask) in zip(problem.markets, problem.markets_masks)
        probabilities_market = @view probabilities[mask, :]
        jacobian_market = @view jacobian[mask, :]

        jacobian_shares_by_theta2!(market, probabilities_market, jacobian_market)
    end
end

