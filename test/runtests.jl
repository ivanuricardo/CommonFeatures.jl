using CommonFeatures
using Test
using LinearAlgebra, Statistics, Random, TensorToolbox, Distributions, Zygote

include("./Regressions/test-infocrit.jl")
include("./Regressions/test-matrixar.jl")
include("./Regressions/test-regutils.jl")
include("./Regressions/test-grads.jl")
include("./Regressions/test-tuckerreg.jl")
include("./SimFunctions/test-simtucker.jl")
