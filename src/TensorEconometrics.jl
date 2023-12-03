module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra

export tensorols, art
export tlag, ridgerankselect
export makecompanion, isstable
export infocrit, tuckerpar
export tuckerreg, clipgradient!
include("./Regressions/MatrixAR.jl")
include("./Regressions/utils.jl")
include("./Regressions/companionmatrix.jl")
include("./Regressions/infocrit.jl")
include("./Regressions/tuckerreg.jl")

export simulatetuckerdata, simulatemardata, simstats
include("./SimFunctions/simfunctions.jl")

end
