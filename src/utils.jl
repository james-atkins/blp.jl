
"""
	tril_indices

Return the (linear) indices of a lower triangle matrix.
"""
function tril_indices(A::AbstractMatrix)
    n = size(A, 1)
    linear_indices = LinearIndices(A)
    low_tri_indices = Vector{Int64}(undef, 0)
    sizehint!(low_tri_indices, div(n * (n + 1), 2))

    for idx in eachindex(A)
        # Ignore upper triangle part
        if idx[2] > idx[1]
            continue
        end
        push!(low_tri_indices, linear_indices[idx])
    end

    return low_tri_indices
end


function chol_tcrossprod(A)
    _, R = qr(A)
    return cholesky(R' * R)
end


"""Simple model for generalised instrumental variables estimation."""
struct IVGMM{T <: AbstractFloat}
    X::Matrix{T}
    S::Matrix{T}

    function IVGMM(X::AbstractMatrix{T}, Z::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T}
        # The solution β̂ to an IV-GMM solves (X' Z W Z' X)β̂ = (X' Z W Z')y
        # This can be solved by letting W = U'U be the Cholesky decomposition of W and letting
        # QR = UZ'X be the thin QR decomposition of UZ'X. Then the above equation simplifies to
        # Rβ̂ = Q'UZ'y => β̂ = Sy which is easy to solve as R is upper triangular.
        W_chol = cholesky(W)
        Q, R = qr(W_chol.U * Z' * X)
        S = UpperTriangular(R) \ Matrix(Q)' * W_chol.U * Z'

        new{T}(X, S)
    end
end

function estimate(ivgmm::IVGMM{T}, y::AbstractVector{T}) where {T}
    return ivgmm.S * y
end

function residuals!(ivgmm::IVGMM, y, residuals)
    beta_hat = estimate(ivgmm, y)
    residuals .= y
    mul!(residuals, ivgmm.X, beta_hat, -1, 1)
end

function residuals_jacobian!(ivgmm::IVGMM, jacobian)
    N = size(ivgmm.X, 1)
    @inbounds for col_idx = 1:N
        @inbounds for row_idx = 1:N
            if row_idx == col_idx
                jacobian[row_idx, col_idx] = 1.0
            else
                jacobian[row_idx, col_idx] = 0.0
            end
        end
    end
    mul!(jacobian, ivgmm.X, ivgmm.S, -1, 1)
end

function ivgmm(X::AbstractMatrix, Z::AbstractMatrix, W::AbstractMatrix, y::AbstractVector)
    return estimate(IVGMM(X, Z, W), y)
end
