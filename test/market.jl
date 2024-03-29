@testset "BLP inversion: $it" for it in (SimpleFixedPointIteration(), SQUAREMIteration(), NLSolveInversion())
    # This random seed gives enough of an outside option so the contraction can swiftly converge
    Random.seed!(123)

    J = 50
    I = 1000

    sigma = [2.0 0.0; 4.0 1.0]
    K2 = size(sigma, 1)
    theta2 = Theta2(sigma)

    x2 = exp.(randn((J, K2)))
    nodes = randn((I, K2))
    weights = fill(1 / I, I)

    delta = exp.(randn(J))
    mu = x2 * sigma * nodes'

    probs = choice_probabilities(delta .+ mu)
    shares = probs * weights

    market = Market(shares, x2, weights, nodes)

    result, solved_shares, solved_probs = solve_demand(market, theta2, it)

    @test result.status == INVERSION_CONVERGED
    @test result.delta ≈ delta
    @test solved_shares ≈ shares
    @test solved_probs ≈ probs
end

@testset "Jacobians of market shares" begin
    Random.seed!(123)
    sigma = [2.0 0.0; 4.0 1.0]

    J = 50
    I = 1000
    K2 = size(sigma, 1)

    theta2 = Theta2(sigma)

    x2 = exp.(randn((J, K2)))
    tastes = randn((I, K2))
    weights = fill(1 / I, I)

    delta = exp.(randn(J))
    mu = x2 * sigma * tastes'

    probs = choice_probabilities(delta .+ mu)
    shares = probs * weights
    market = Market(shares, x2, weights, tastes)

    function compute_shares(s, d)
        mu = x2 * ((s * tastes'))
        return choice_probabilities(d .+ mu) * weights
    end

    shares_theta2_jacobian = BLP.jacobian_shares_by_theta2(market, probs)
    shares_theta2_jacobian_fd = finite_difference_jacobian(s -> compute_shares(s, delta), sigma)

    # The analytic gradient is calculated with respect to the elements of the lower triangle of sigma but the finite
    # differences gradient is wrt to all elements of sigma. So we need to remove the columns of the finite differences
    # gradient corresponding to the upper triangular part of sigma.
    fd_indices = BLP.tril_indices(theta2.sigma)
    @test isapprox(shares_theta2_jacobian, shares_theta2_jacobian_fd[:, fd_indices], rtol = 1E-6)
    @test eltype(shares_theta2_jacobian) != Any

    shares_delta_jacobian = BLP.jacobian_shares_by_delta(market, probs)
    shares_delta_jacobian_fd = finite_difference_jacobian(d -> compute_shares(sigma, d), delta)

    @test isapprox(shares_delta_jacobian, shares_delta_jacobian_fd, rtol = 1E-6)
    @test eltype(shares_delta_jacobian) != Any
end

@testset "Jacobian of shares wrt price" begin
    Random.seed!(123)
    theta2 = Theta2(-0.5, fill(-1.2, 1, 1))

    J = 50
    I = 1000
    K2 = size(theta2.sigma, 1)

    prices = exp.(randn((J)))
    tastes = randn((I, K2))
    weights = fill(1 / I, I)

    mu = prices * theta2.sigma * tastes'
    xi = exp.(randn(J))
    probs = choice_probabilities(xi .+ (prices * theta2.alpha) .+ mu)
    market = Market(probs * weights, reshape(prices, :, 1), weights, tastes)

    function shares_price(prices)
        market_new = Market(market.shares, reshape(prices, :, 1), market.weights, market.tastes)
        delta = xi + (theta2.alpha * prices)

        shares, probs = BLP.compute_shares_and_choice_probabilities(market_new, delta, theta2)

        return shares
    end

    shares_prices_jacobian_fd = finite_difference_jacobian(p -> shares_price(p), prices)
    shares_prices_jacobian = BLP.jacobian_shares_by_price(market, probs, theta2, 1)
    @test isapprox(shares_prices_jacobian, shares_prices_jacobian_fd, rtol = 1E-5)
    @test isapprox(shares_prices_jacobian, shares_prices_jacobian')
end
