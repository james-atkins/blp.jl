export choice_probabilities, choice_probabilities!

"""
Compute the choice probabilities using an overflow safe algorithm.

This is just a wrapper around [`choice_probabilities!`](@ref) which computes them in-place.
"""
function choice_probabilities(utilities::AbstractMatrix{<:AbstractFloat})
    probabilities = similar(utilities)
    choice_probabilities!(probabilities, utilities)
    return probabilities
end

"""
Compute the choice probabilities in-place using an overflow safe algorithm.

Based on the SoftmaxThreePassReload function from
Dukhan and Ablavatski (2020), The Two-Pass Softmax Algorithm (https://arxiv.org/abs/2001.04438).
"""
function choice_probabilities!(output::AbstractMatrix{T}, utilities::AbstractMatrix{T}) where T<:AbstractFloat
    @assert size(output) == size(utilities) "Output matrix must have same size as utilities matrix."
    J, I = size(utilities)

    @inbounds for i = 1:I
        max_u = T(0)
        for j = 1:J
            if utilities[j, i] > max_u
                max_u = utilities[j, i]
            end
        end

        sigma = @fastmath exp(-max_u)

        @simd ivdep for j = 1:J
            output[j, i] = @fastmath exp(utilities[j, i] - max_u)
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
