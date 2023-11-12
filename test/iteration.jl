# Test that an example fixed point problem converges to the exact solution. Based on the pyblp code.
@testset "Iteration converges $it" for it in (SimpleFixedPointIteration(), SQUAREMIteration())
    function contraction!(x_out::AbstractVector, x::AbstractVector)
        c1 = [10, 12]
        c2 = [3, 5]

        @. x_out = sqrt(c1 / (x + c2))
        return x_out
    end

    initial = ones(2)
    result = fixed_point_iteration(it, initial, contraction!)

    @test result.status == INVERSION_CONVERGED
    @test result.delta ≈ [1.4920333, 1.37228132]
end

# Test that the solution to the fixed point problem from Hasselblad (1969) is reasonably close to the exact
# solution. This same problem is used in an original SQUAREM unit test.
@testset "SQUAREM $scheme Hasselblad" for scheme in (SQUAREM1, SQUAREM2, SQUAREM3)
    function contraction!(x_out::AbstractVector, x::AbstractVector)
        y = [162, 267, 271, 185, 111, 61, 27, 8, 3, 1]
        i = 0:length(y)-1
        z = @. (x[1] * exp(-x[2]) * x[2]^i) / (x[1] * exp(-x[2]) * x[2]^i + (1 - x[1]) * exp(-x[3]) * x[3]^i)

        x_out[:] = @. [$sum(y * z) / $sum(y), $sum(y * i * z) / $sum(y .* z), $sum(y * i * (1 - z)) / $sum(y * (1 - z))]
        return x_out
    end

    it = SQUAREMIteration(scheme = scheme)

    initial = [0.2, 2.5, 1.5]

    result = fixed_point_iteration(it, initial, contraction!)
    @test result.status == INVERSION_CONVERGED
    @test result.delta ≈ [0.6401146029910, 2.6634043566619, 1.2560951012662]
end
