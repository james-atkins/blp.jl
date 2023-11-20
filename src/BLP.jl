module BLP

using BlockDiagonals: BlockDiagonal
import LinearAlgebra as LA
using LinearAlgebra: Diagonal, LowerTriangular, Symmetric, ⋅, mul!, rank, cholesky
using NLsolve: IsFiniteException, nlsolve, converged, only_fj!

include("utils.jl")

@enum InversionStatus begin
    INVERSION_CONVERGED
    INVERSION_NUMERICAL_ISSUES
    INVERSION_EXCEEDED_MAX_ITERATIONS
end

struct InversionResult{T <: AbstractFloat}
    status::InversionStatus
    delta::Vector{T}
    iterations::Int
    evaluations::Int
end

export InversionResult,
    InversionStatus, INVERSION_CONVERGED, INVERSION_NUMERICAL_ISSUES, INVERSION_EXCEEDED_MAX_ITERATIONS

abstract type Iteration end
include("iteration.jl")
export Iteration, fixed_point_iteration, SimpleFixedPointIteration, SQUAREMIteration, SQUAREM1, SQUAREM2, SQUAREM3

include("softmax.jl")
export choice_probabilities, choice_probabilities!

include("theta2.jl")
export Theta2, flatten

include("market.jl")
export Market, NLSolveInversion, compute_mu, solve_demand

include("problem.jl")
export Problem, Products, Individuals, compute_shares_and_choice_probabilities!, jacobian_shares_by_delta!, jacobian_shares_by_theta2!

end
