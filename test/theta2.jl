@testset "flatten/unflatten Theta2 with sigma" begin
    K2 = 5

    sigma = randn(K2, K2)

    theta2 = Theta2(sigma)
    flat, unflatten = flatten(theta2)

    theta2_unflattened = unflatten(flat)

    @test theta2.sigma == theta2_unflattened.sigma
end

@testset "flatten/unflatten Theta2 with sigma, pi" begin
    K2, D = 4, 3

    sigma = randn(K2, K2)
    pi = randn(K2, D)

    theta2 = Theta2(sigma, pi)
    flat, unflatten = flatten(theta2)

    theta2_unflattened = unflatten(flat)

    @test theta2.sigma == theta2_unflattened.sigma
    @test theta2.pi == theta2_unflattened.pi
end

@testset "flatten/unflatten Theta2 with alpha, sigma" begin
    K2 = 5

    sigma = randn(K2, K2)
    alpha = randn()

    theta2 = Theta2(alpha, sigma)
    flat, unflatten = flatten(theta2)

    theta2_unflattened = unflatten(flat)

    @test theta2.alpha == theta2_unflattened.alpha
    @test theta2.sigma == theta2_unflattened.sigma
end

@testset "flatten/unflatten Theta2 with alpha, sigma, pi" begin
    K2, D = 4, 3

    sigma = randn(K2, K2)
    pi = randn(K2, D)
    alpha = randn()

    theta2 = Theta2(alpha, sigma, pi)
    flat, unflatten = flatten(theta2)

    theta2_unflattened = unflatten(flat)

    @test theta2.alpha == theta2_unflattened.alpha
    @test theta2.sigma == theta2_unflattened.sigma
    @test theta2.pi == theta2_unflattened.pi
end

@testset "flatten/unflatten Theta2 with custom names" begin
    K2, D = 2, 3

    sigma = NamedArray(randn(K2, K2))
    setdimnames!(sigma, ["characteristics", "characteristics"])
    setnames!(sigma, ["sugar", "mushy"], 1)
    setnames!(sigma, ["sugar", "mushy"], 2)

    pi = NamedArray(randn(K2, D))
    setdimnames!(pi, ["characteristics", "demographics"])
    setnames!(pi, ["sugar", "mushy"], 1)
    setnames!(pi, ["gender", "income", "education"], 2)

    alpha = randn()

    theta2 = Theta2(alpha, sigma, pi)
    flat, unflatten = flatten(theta2)

    theta2_unflattened = unflatten(flat)

    @test theta2.alpha == theta2_unflattened.alpha
    @test theta2.sigma == theta2_unflattened.sigma
    @test theta2.pi == theta2_unflattened.pi
    @test names(theta2.sigma) == names(theta2_unflattened.sigma)
    @test names(theta2.pi) == names(theta2_unflattened.pi)
end
