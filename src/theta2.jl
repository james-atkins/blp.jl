struct Theta2{T <: AbstractFloat}
    alpha::Union{T, Nothing}                              # Linear price coefficient if not concentrated out
    sigma::NamedMatrix{T, LowerTriangular{T, Matrix{T}}}  # K2 x K2 lower triangle
    pi::NamedMatrix{T}                                    # K2 x D

    function Theta2(alpha::AbstractFloat, sigma::NamedMatrix, pi::NamedMatrix)
        T = promote_type(typeof(alpha), eltype(sigma), eltype(pi))

        if size(sigma, 1) != size(sigma, 1)
            throw(DimensionMismatch("sigma and pi must have the same number of rows"))
        end

        if names(sigma, 1) != names(sigma, 2)
            throw(DimensionMismatch("sigma must have the same characteristic names"))
        end

        if names(sigma, 1) != names(pi, 1)
            throw(DimensionMismatch("sigma and pi must have the same characteristic names"))
        end

        sigma_parent = parent(sigma)
        sigma_low_tri_named = NamedArray(LowerTriangular(sigma_parent))
        setnames!(sigma_low_tri_named, names(sigma, 1), 1)
        setnames!(sigma_low_tri_named, names(sigma, 2), 2)

        new{T}(alpha, sigma_low_tri_named, pi)
    end

    function Theta2(sigma::NamedMatrix, pi::NamedMatrix)
        T = promote_type(eltype(sigma), eltype(pi))

        if size(sigma, 1) != size(sigma, 1)
            throw(DimensionMismatch("sigma and pi must have the same number of rows"))
        end

        if names(sigma, 1) != names(pi, 1)
            throw(DimensionMismatch("sigma and pi must have the same characteristic names"))
        end

        sigma_parent = parent(sigma)
        sigma_low_tri_named = NamedArray(LowerTriangular(sigma_parent))
        setnames!(sigma_low_tri_named, names(sigma, 1), 1)
        setnames!(sigma_low_tri_named, names(sigma, 2), 2)

        new{T}(nothing, sigma_low_tri_named, pi)
    end
end

function Theta2(sigma::NamedMatrix)
    K2 = size(sigma, 1)
    pi = Matrix{eltype(sigma)}(undef, K2, 0)
    pi_named = NamedArray(pi)
    setnames!(pi_named, names(sigma, 1), 1)

    return Theta2(sigma, pi_named)
end

function Theta2(alpha::AbstractFloat, sigma::NamedMatrix)
    K2 = size(sigma, 1)
    pi = Matrix{eltype(sigma)}(undef, K2, 0)
    pi_named = NamedArray(pi)
    setnames!(pi_named, names(sigma, 1), 1)

    return Theta2(alpha, sigma, pi_named)
end

function Theta2(sigma::AbstractMatrix)
    sigma_named = NamedArray(sigma)
    setdimnames!(sigma_named, ["characteristics", "characteristics"])
    return Theta2(sigma_named)
end

function Theta2(sigma::AbstractMatrix, pi::AbstractMatrix)
    sigma_named = NamedArray(sigma)
    setdimnames!(sigma_named, ["characteristics", "characteristics"])

    pi_named = NamedArray(pi)
    setdimnames!(pi_named, ["characteristics", "demographics"])

    return Theta2(sigma_named, pi_named)
end

function Theta2(alpha::AbstractFloat, sigma::AbstractMatrix)
    sigma_named = NamedArray(sigma)
    setdimnames!(sigma_named, ["characteristics", "characteristics"])
    return Theta2(alpha, sigma_named)
end

function Theta2(alpha::AbstractFloat, sigma::AbstractMatrix, pi::AbstractMatrix)
    sigma_named = NamedArray(sigma)
    setdimnames!(sigma_named, ["characteristics", "characteristics"])

    pi_named = NamedArray(pi)
    setdimnames!(pi_named, ["characteristics", "demographics"])

    return Theta2(alpha, sigma_named, pi_named)
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

    characteristic_names = names(theta2.sigma, 1)
    demographic_names = names(theta2.pi, 2)

    function unflatten(x)
        if length(x) != n
            throw(DimensionMismatch("incorrect length for flat Theta2"))
        end

        sigma = NamedArray(LowerTriangular(zeros(T, K2, K2)))
        sigma[tril_indices(sigma)] = x[sigma_idx_1:sigma_idx_2]
        setnames!(sigma, characteristic_names, 1)
        setnames!(sigma, characteristic_names, 2)

        pi = NamedArray(Matrix{T}(undef, K2, D))
        if D > 0
            vec(pi)[:] = x[pi_idx_1:pi_idx_2]
        end
        setnames!(pi, characteristic_names, 1)
        setnames!(pi, demographic_names, 2)

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
