@testset "BLP inversion: $it" for it in (SimpleFixedPointIteration(), SQUAREMIteration())
    # This random seed gives enough of an outside option so the contraction can swiftly converge
    Random.seed!(123)

    J = 50
    I = 1000

    sigma = [2.0 0.0; 4.0 1.0]
    K2 = size(sigma, 1)
    theta2 = Theta2(sigma)

    x2 = exp.(randn((J, K2)))
    nodes = randn((I, K2))
    weights = fill(1/I, I)

    delta = exp.(randn(J))
    mu = x2 * sigma * nodes'

    shares = choice_probabilities(delta .+ mu) * weights

    market = Market(shares, x2, weights, nodes)

    result = compute_delta(market, theta2, it)

    @test result.status == ITERATION_CONVERGED
    @test result.delta ≈ delta
end
