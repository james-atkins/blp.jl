@testset "flatten/unflatten Theta2 without pi" begin
    K2 = 5

    sigma = randn(K2, K2)

    theta2 = Theta2(sigma)
    flat, unflatten = flatten(theta2)

    theta2_unflattened = unflatten(flat)

    @test theta2.sigma == theta2_unflattened.sigma
end

@testset "flatten/unflatten Theta2" begin
    K2, D = 4, 3

    sigma = randn(K2, K2)
    pi = randn(K2, D)

    theta2 = Theta2(sigma, pi)
    flat, unflatten = flatten(theta2)

    theta2_unflattened = unflatten(flat)

    @test theta2.sigma == theta2_unflattened.sigma
    @test theta2.pi == theta2_unflattened.pi
end
