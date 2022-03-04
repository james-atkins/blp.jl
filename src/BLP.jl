module BLP

import LinearAlgebra as LA
using LinearAlgebra: LowerTriangular, ⋅, mul!

include("iteration.jl")
include("softmax.jl")
include("market.jl")

end
