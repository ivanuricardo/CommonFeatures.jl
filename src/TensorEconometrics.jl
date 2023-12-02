module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra

export tensorols
export tlag
include("MatrixAR/MatrixAR.jl")
include("./MatrixAR/utils.jl")


end
