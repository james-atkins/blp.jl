function compute_blp_instruments(data)
	K = size(data.X1, 2)
	J = data.J
	M = data.M
	chars_array = reshape(data.X1, J, M, K)

	# Cannot sum over over other products produced by the firm as single product
	own_chars = data.X1
	blp_rivals = reshape(sum(chars_array, dims=1) .- chars_array, J * M, K)

	# Drop colinear column
    blp_rivals = blp_rivals[:, 2:end]
	return [own_chars blp_rivals]
end

products = begin
    M, J = 100, 3
    raw_data = matread("gaston.mat")

    data = (
        J = J,
        M = M,
    	product_ids = repeat(1:J, outer = M),
    	market_ids = repeat(1:M, inner = J),
    	shares = reshape(raw_data["shares"], :),
        X1 = raw_data["x1"],
    	prices = reshape(raw_data["P_opt"], :),
	)

    X2 = reshape(data.prices, :, 1)
	Z_demand = compute_blp_instruments(data)

	Products(
		market_ids = data.market_ids,
		shares = data.shares,
		prices = data.prices,
		X1_exog = data.X1,
		X2 = X2,
		Z_demand = Z_demand
	)
end

individuals = begin
    I = 1_000
    M = 100
	market_ids = repeat(1:M, inner = I)
	weights = fill(1/I, I * M)
    tastes = exp.(randn(I * M, products.K2))

	Individuals(market_ids, weights, tastes)
end

@testset "Jacobians of market shares for all markets" begin
    Random.seed!(123)

    problem = Problem(products, individuals)
    sigma = fill(0.5, 1, 1)
    theta2 = Theta2(sigma)
    delta = 10 .* randn(problem.N)

    @test BLP.check_compatible_theta2(problem, theta2)

    function compute_shares(d, s)
        theta2 = Theta2(s)

        T = eltype(problem)
        shares = Vector{T}(undef, problem.N)
        probabilities = [Matrix{T}(undef, market.J, market.I) for market in problem.markets]

        compute_shares_and_choice_probabilities!(problem, d, theta2, shares, probabilities)
    	return shares
    end

    function compute_jacobian_shares_delta(d)
    	shares, probs = BLP.compute_shares_and_choice_probabilities(problem, d, theta2)

    	blocks = map(m -> Matrix{eltype(problem)}(undef, m.J, m.J), problem.markets)
        jacobian = BlockDiagonal(blocks)

    	jacobian_shares_by_delta!(problem, shares, probs, jacobian)
    	return jacobian
    end

    function compute_jacobian_shares_theta2(s)
        theta2 = Theta2(s)
    	_, probabilities = BLP.compute_shares_and_choice_probabilities(problem, delta, theta2)

        P = div(problem.K2 * (problem.K2 + 1), 2) + (problem.K2 * problem.D)
        jacobian = Matrix{eltype(problem)}(undef, problem.N, P)

        jacobian_shares_by_theta2!(problem, probabilities, jacobian)
        return jacobian
    end

    jacobian_shares_delta = compute_jacobian_shares_delta(delta)
    fd_jacobian_shares_delta = finite_difference_jacobian(d -> compute_shares(d, sigma), delta)
    @test isapprox(fd_jacobian_shares_delta, jacobian_shares_delta, rtol = 1E-6)
    @test eltype(jacobian_shares_delta) != Any

    jacobian_shares_theta2 = compute_jacobian_shares_theta2(sigma)
    fd_jacobian_shares_theta2 = finite_difference_jacobian(s -> compute_shares(delta, s), sigma)
    @test isapprox(fd_jacobian_shares_theta2, jacobian_shares_theta2, rtol = 1E-6)
    @test eltype(jacobian_shares_theta2) != Any
end
