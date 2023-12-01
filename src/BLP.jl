module BLP

using BlockDiagonals: BlockDiagonal
import LinearAlgebra as LA
using LinearAlgebra: Diagonal, Symmetric, LowerTriangular, UpperTriangular, Cholesky, ⋅, mul!, rank, cholesky, qr
using NLsolve: IsFiniteException, nlsolve, converged, only_fj!
using Statistics: mean

import GMM:
    GMMModel,
    solve,
    gmm_success,
    gmm_estimate,
    gmm_moments,
    gmm_num_residuals,
    gmm_num_instruments,
    gmm_num_constraints,
    gmm_num_parameters,
    gmm_instruments,
    gmm_residuals_constraints!,
    gmm_residuals_constraints_jacobians!,
    gmm_success,
    gmm_objective_value,
    gmm_estimate,
    gmm_moments,
    gmm_moments_jacobian,
    gmm_constraints_jacobian


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
export Problem,
    Products,
    Individuals,
    compute_shares_and_choice_probabilities!,
    jacobian_shares_by_delta!,
    jacobian_shares_by_theta2!

include("mpec.jl")
export solve_mpec

end
