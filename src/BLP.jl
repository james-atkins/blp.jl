module BLP

import LinearAlgebra as LA
using LinearAlgebra: Diagonal, LowerTriangular, ⋅, mul!
using NLsolve

include("iteration.jl")
include("softmax.jl")
include("market.jl")

end
