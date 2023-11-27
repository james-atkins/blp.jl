struct Theta2{T <: AbstractFloat}
    alpha::Union{T, Nothing}              # Linear price coefficient if not concentrated out
    sigma::LowerTriangular{T, Matrix{T}}  # K2 x K2 lower triangle
    pi::Matrix{T}                         # K2 x D

    function Theta2(alpha::AbstractFloat, sigma::LowerTriangular, pi::Matrix)
        T = promote_type(typeof(alpha), eltype(sigma), eltype(pi))

        if size(sigma, 1) != size(sigma, 1)
            throw(DimensionMismatch("sigma and pi must have the same number of rows"))
        end

        new{T}(alpha, sigma, pi)
    end

    function Theta2(sigma::LowerTriangular, pi::Matrix)
        T = promote_type(eltype(sigma), eltype(pi))

        if size(sigma, 1) != size(sigma, 1)
            throw(DimensionMismatch("sigma and pi must have the same number of rows"))
        end

        new{T}(nothing, sigma, pi)
    end
end

function Theta2(sigma::AbstractMatrix, pi::AbstractMatrix)
    return Theta2(LowerTriangular(sigma), pi)
end

function Theta2(sigma::AbstractMatrix)
    K2 = size(sigma, 1)
    pi = Matrix{eltype(sigma)}(undef, K2, 0)
    return Theta2(LowerTriangular(sigma), pi)
end

function Theta2(alpha::AbstractFloat, sigma::AbstractMatrix)
    K2 = size(sigma, 1)
    pi = Matrix{eltype(sigma)}(undef, K2, 0)
    return Theta2(alpha, LowerTriangular(sigma), pi)
end

function Theta2(alpha::AbstractFloat, sigma::AbstractMatrix, pi::AbstractMatrix)
    return Theta2(alpha, LowerTriangular(sigma), pi)
end

Base.eltype(::Type{Theta2{T}}) where {T} = T

function flatten(theta2::Theta2)
    K2, D = size(theta2.pi)

    n = div(K2 * (K2 + 1), 2) + (K2 * D)
    is_alpha = theta2.alpha !== nothing
    if is_alpha
        n += 1
    end

    sigma_idx_1 = 1
    sigma_idx_2 = div(K2 * (K2 + 1), 2)

    if D > 0
        pi_idx_1 = sigma_idx_2 + 1
        pi_idx_2 = pi_idx_1 + (K2 * D) - 1
    else
        pi_idx_1 = pi_idx_2 = 0
    end

    T = eltype(theta2)

    function unflatten(x)
        if length(x) != n
            throw(DimensionMismatch("incorrect length for flat Theta2"))
        end

        sigma = LowerTriangular(Matrix{T}(undef, K2, K2))
        sigma[tril_indices(sigma)] = x[sigma_idx_1:sigma_idx_2]

        pi = Matrix{T}(undef, K2, D)
        if D > 0
            vec(pi)[:] = x[pi_idx_1:pi_idx_2]
        end

        if is_alpha
            alpha = x[length(x)]
            return Theta2(alpha, sigma, pi)
        else
            return Theta2(sigma, pi)
        end
    end

    if is_alpha
        return ([theta2.sigma[tril_indices(theta2.sigma)]; vec(theta2.pi); theta2.alpha], unflatten)
    else
        return ([theta2.sigma[tril_indices(theta2.sigma)]; vec(theta2.pi)], unflatten)
    end
end
