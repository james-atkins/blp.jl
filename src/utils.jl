
"""
	tril_indices

Return the (linear) indices of a lower triangle matrix.
"""
function tril_indices(A::LowerTriangular)
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
