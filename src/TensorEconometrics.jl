module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra

export tensorols, art
export tlag, ridgerankselect
export makecompanion, isstable
include("./Regressions/MatrixAR.jl")
include("./Regressions/utils.jl")
include("./Regressions/companionmatrix.jl")


end
