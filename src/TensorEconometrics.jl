module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra

export tensorols
export tlag
include("./Regressions/MatrixAR.jl")
include("./Regressions/utils.jl")


end
