module BLP

import LinearAlgebra as LA
using LinearAlgebra: Diagonal, LowerTriangular, ⋅, mul!
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

end
