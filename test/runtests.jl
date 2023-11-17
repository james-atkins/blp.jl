using BLP
using Test
import Random
using FiniteDiff: finite_difference_jacobian
using LinearAlgebra: LowerTriangular

include("theta2.jl")
include("iteration.jl")
include("market.jl")
