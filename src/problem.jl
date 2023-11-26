struct Products{T <: AbstractFloat}
    market_ids::Vector{Int}  # N
    shares::Vector{T}        # N
    prices::Vector{T}        # N
    X1_exog::Matrix{T}       # N x K1_exog
    X2::Matrix{T}            # N x K2
    Z_demand::Matrix{T}      # N x M_demand

    market_masks::Vector{BitVector}

    N::Int32         # Number of products
    K1_exog::Int32   # Number of exogenous demand-side linear product characteristics
    K2::Int32        # Number of demand-side nonlinear product characteristics
    M_demand::Int32  # Number of demand-side instruments

    function Products(; market_ids, shares, prices, X1_exog, X2, Z_demand)
        # Check dimensions
        if !allequal([size(x, 1) for x in [shares, prices, X1_exog, X2, Z_demand]])
            throw(DimensionMismatch())
        end

        # Rank condition for instruments
        X1 = [X1_exog prices]
        z_x1 = Z_demand' * X1
        if rank(z_x1) < min(size(z_x1)...)
            error("Matrix Z' X₁ does not have full rank.")
        end

        # Split into markets
        market_masks = begin
            if !issorted(market_ids)
                error("Products must be sorted by market")
            end

            unique_market_ids = unique(market_ids)
            masks = Vector{BitVector}(undef, length(unique_market_ids))
            for (idx, id) in enumerate(unique_market_ids)
                mask = market_ids .== id
                masks[idx] = mask
            end

            masks
        end

        T = promote_type(eltype(shares), eltype(prices), eltype(X1_exog), eltype(X2), eltype(Z_demand))
        N = size(shares, 1)
        K1_exog = size(X1_exog, 2)
        K2 = size(X2, 2)
        M_demand = size(Z_demand, 2)

        new{T}(
            market_ids,
            shares,
            prices,
            X1_exog,
            X2,
            Z_demand,

            market_masks,

            N,
            K1_exog,
            K2,
            M_demand
        )
    end
end

Base.eltype(::Type{Products{T}}) where {T} = T


struct Individuals{T <: AbstractFloat}
    market_ids::Vector{Int}  # I
    weights::Vector{T}       # I
    tastes::Matrix{T}        # I x K2
    demographics::Matrix{T}  # I x D

    market_masks::Vector{BitVector}

    K2::Int32        # Number of demand-side nonlinear product characteristics
    I::Int32         # Number of individuals across all markets
    D::Int32         # Number of demographic variables

    function Individuals(market_ids, weights, tastes, demographics)
        # Check dimensions
        if !allequal([size(x, 1) for x in [market_ids, weights, tastes, demographics]])
            throw(DimensionMismatch())
        end

        # Split into markets
        market_masks = begin
            if !issorted(market_ids)
                error("Products must be sorted by market")
            end

            unique_market_ids = unique(market_ids)
            masks = Vector{BitVector}(undef, length(unique_market_ids))
            for (idx, id) in enumerate(unique_market_ids)
                mask = market_ids .== id
                masks[idx] = mask
            end

            masks
        end

        T = promote_type(eltype(weights), eltype(tastes), eltype(demographics))
        I = length(weights)
        K2 = size(tastes, 2)
        D = size(demographics, 2)

        new{T}(
            market_ids,
            weights,
            tastes,
            demographics,

            market_masks,

            K2,
            I,
            D
        )
    end
end


function Individuals(market_ids, weights, tastes)
    demographics = Matrix{eltype(tastes)}(undef, size(tastes, 1), 0)
    return Individuals(market_ids, weights, tastes, demographics)
end

Base.eltype(::Type{Individuals{T}}) where {T} = T


struct Problem{T <: AbstractFloat}
    products::Products{T}
    individuals::Individuals{T}
    
    markets::Vector{Market}

    N::Int32         # Number of products
    K1_exog::Int32   # Number of demand-side linear product characteristics
    K2::Int32        # Number of demand-side nonlinear product characteristics
    M_demand::Int32  # Number of demand-side instruments
    I::Int32         # Number of individuals
    D::Int32         # Number of demographic variables

    function Problem(products::Products{T}, individuals::Individuals{T}) where {T}
        if products.K2 != individuals.K2
            throw(DimensionMismatch())
        end

        if length(products.market_masks) != length(individuals.market_masks)
            throw(DimensionMismatch("The number of markets must be the same for products and individuals"))
        end

        markets = Vector{Market}(undef, 0)
        sizehint!(markets, length(products.market_masks))

        for (pmask, imask) in zip(products.market_masks, individuals.market_masks)
            market = Market(
                products.shares[pmask],
                products.X2[pmask, :],
                individuals.weights[imask],
                individuals.tastes[imask, :],
                individuals.demographics[imask, :]
            )
            push!(markets, market)
        end

        new{T}(
            products,
            individuals,

            markets,

            products.N,
            products.K1_exog,
            products.K2,
            products.M_demand,
            individuals.I,
            individuals.D
        )
    end
end

Base.eltype(::Type{Problem{T}}) where {T} = T


function check_compatible_theta2(problem::Problem, theta2::Theta2)
    if size(theta2.sigma) != (problem.K2, problem.K2)
        throw(DimensionMismatch("sigma has incompatible size. Expected $((problem.K2, problem.K2)) not $(size(theta2.sigma))"))
    end

    if size(theta2.pi) != (problem.K2, problem.D)
        throw(DimensionMismatch("pi has incompatible size. Expected $((problem.K2, problem.D)) not $(size(theta2.pi))"))
    end

    return true
end


"""
Given guesses of `delta` and `theta2`, compute the choice probabilities and market shares.
"""
function compute_shares_and_choice_probabilities!(problem::Problem, delta::AbstractVector, theta2::Theta2, shares::AbstractVector, probabilities::AbstractVector{<: AbstractMatrix})
    if length(shares) != problem.N
        throw(DimensionMismatch("shares has invalid length. expected $(problem.N); actual $(length(shares))"))
    end

    if length(probabilities) != length(problem.markets)
        throw(DimensionMismatch("probabilities has invalid length. expected: $(length(problem.markets)); actual: $(length(probabilities))"))
    end

    for (market, mask, probabilities_market) in zip(problem.markets, problem.products.market_masks, probabilities)
        delta_market = @view delta[mask]
        shares_market = @view shares[mask]

        compute_shares_and_choice_probabilities!(market, delta_market, theta2, shares_market, probabilities_market)
    end
end

function compute_shares_and_choice_probabilities(problem::Problem, delta::AbstractVector, theta2::Theta2)
    T = eltype(problem)
    shares = Vector{T}(undef, problem.N)
    probabilities = [Matrix{T}(undef, market.J, market.I) for market in problem.markets]

    compute_shares_and_choice_probabilities!(problem, delta, theta2, shares, probabilities)
    return (shares, probabilities)
end


"""
Solve for the mean utility that equates observed and predicted market shares.
"""
function solve_demand(problem::Problem, theta2::Theta2, method; tolerance = 1E-14)
    results = Vector{InversionResult}(undef, length(problem.markets))
    probabilities = Vector{Matrix{eltype(problem)}}(undef, length(problem.markets))

    Threads.@threads for idx in 1:length(problem.markets)
        market = problem.markets[idx]
        result, _, probs_market = solve_demand(market, theta2, method; tolerance = tolerance)
        results[idx] = result
        probabilities[idx] = probs_market
    end

    for result in results
        if result.status != INVERSION_CONVERGED
            @info "Market did not converge: $(result.status)"
        end
    end

    delta = mapreduce(r -> r.delta, vcat, results)
    return delta, probabilities
end


function jacobian_shares_by_delta!(problem::Problem, shares::AbstractVector, probabilities::AbstractVector{<: AbstractMatrix}, jacobian::BlockDiagonal)
    if length(shares) != problem.N
        throw(DimensionMismatch("shares has invalid length. expected $(problem.N); actual $(length(shares))"))
    end

    if length(probabilities) != length(problem.markets)
        throw(DimensionMismatch("probabilities has invalid length. expected: $(length(problem.markets)); actual: $(length(probabilities))"))
    end

    if length(problem.markets) != length(jacobian.blocks)
        throw(DimensionMismatch("jacobian has invalid number of blocks. expected $(length(problem.markets)); actual $(length(jacobian.blocks))"))
    end

    for (market, mask, probabilities_market, block) in zip(problem.markets, problem.products.market_masks, probabilities, jacobian.blocks)
        shares_market = @view shares[mask]

        jacobian_shares_by_delta!(market, shares_market, probabilities_market, block)
    end
end


function jacobian_shares_by_theta2!(problem::Problem, probabilities::AbstractVector{<: AbstractMatrix}, jacobian::AbstractMatrix)
    P = div(problem.K2 * (problem.K2 + 1), 2) + (problem.K2 * problem.D)
    if size(jacobian) != (problem.N, P)
        throw(DimensionMismatch("jacobian has invalid size. expected $((problem.N, P)); actual $(size(probabilities))"))
    end

    for (market, mask, probabilities_market) in zip(problem.markets, problem.products.market_masks, probabilities)
        jacobian_market = @view jacobian[mask, :]

        jacobian_shares_by_theta2!(market, probabilities_market, jacobian_market)
    end
end


"""
The problem can be simplified by noting the following:

Let W = U'U be the Cholesky decomposition of W and let QR = UZ'X be the thin QR decomposition of UZ'X.
Then we can form both the quadratic objective and concentrate out the linear parameters at the same time
by using the adjusted weighting matrix W̃ = U'(I - QQ')U.
"""
function make_adjusted_weighting_matrix(problem::Problem, W)
    X1 = [problem.products.X1_exog problem.products.prices]
    Z = problem.products.Z_demand

    # Compute modified weighing matrix W̃ = U'(I - QQ')U that both forms the quadratic GMM objective
    # and concentrates out linear parameters.
    W_chol = cholesky(W)
    Q, _ = qr(W_chol.U * Z' * X1)
    Q_thin = Matrix(Q)
    W_tilde = LowerTriangular(W_chol.L) * Symmetric(LA.I - (Q_thin * Q_thin')) * UpperTriangular(W_chol.U)

    # Enforce symmetry
    @assert isapprox(W_tilde, W_tilde')
    W_tilde = Symmetric(W_tilde .+ W_tilde / 2)

    return W_tilde
end
