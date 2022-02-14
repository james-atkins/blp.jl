
# Based on the SoftmaxThreePassReload function from
# Dukhan and Ablavatski (2020), The Two-Pass Softmax Algorithm (https://arxiv.org/abs/2001.04438)
function choice_probabilities(utilities::Matrix{Float64})
    J, I = size(utilities)
    choice_probs = Matrix{Float64}(undef, J, I)

    @inbounds for i = 1:I
        max_u = 0.0
        for j = 1:J
            if utilities[j, i] > max_u
                max_u = utilities[j, i]
            end
        end

        sigma = @fastmath exp(-max_u)

        @simd ivdep for j = 1:J
            choice_probs[j, i] = @fastmath exp(utilities[j, i] - max_u)
            sigma += choice_probs[j, i]
        end
    
        sigma_inv = 1 / sigma
        @simd ivdep for j = 1:J
            choice_probs[j, i] *= sigma_inv
        end
    end

    return choice_probs
end
