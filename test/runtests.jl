using BLP
using Test
import Random
using BlockDiagonals: BlockDiagonal
using FiniteDiff: finite_difference_jacobian
using LinearAlgebra: LowerTriangular
using MAT: matread

include("theta2.jl")
include("iteration.jl")
include("market.jl")
include("problem.jl")
