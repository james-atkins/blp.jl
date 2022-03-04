export choice_probabilities!

"""
Compute the choice probabilities in-place using an overflow safe algorithm.

Based on the SoftmaxThreePassReload function from
Dukhan and Ablavatski (2020), The Two-Pass Softmax Algorithm (https://arxiv.org/abs/2001.04438).
"""
function choice_probabilities!(output::AbstractMatrix{T}, delta::AbstractVector{T}, mu::AbstractMatrix{T}) where T<:AbstractFloat
    J, I = size(mu)
    @assert length(delta) == J "delta and mu must have the same number of rows"
    @assert size(output) == (J, I) "output matrix must be same shape as mu"

    @inbounds for i = 1:I
        max_u = T(0)
        for j = 1:J
            u = delta[j] + mu[j, i]
            if u > max_u
                max_u = u
            end
        end

        sigma = @fastmath exp(-max_u)

        @simd ivdep for j = 1:J
            output[j, i] = @fastmath exp(delta[j] + mu[j, i] - max_u)
            # SIMD makes left-to-right summation both faster and more accurate
            # https://discourse.julialang.org/t/when-shouldnt-we-use-simd/18276/14
            sigma += output[j, i]
        end
    
        sigma_inv = 1 / sigma
        @simd ivdep for j = 1:J
            output[j, i] *= sigma_inv
        end
    end
end
