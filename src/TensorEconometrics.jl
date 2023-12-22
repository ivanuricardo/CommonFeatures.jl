module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra
using ProgressBars

export tensorols, art
export tlag, ridgerankselect, rescalemat
export makecompanion, isstable
export infocrit, tuckerpar
export tuckerreg, clipgradient!
include("./Regressions/MatrixAR.jl")
include("./Regressions/utils.jl")
include("./Regressions/companionmatrix.jl")
include("./Regressions/infocrit.jl")
include("./Regressions/tuckerreg.jl")

export simulatetuckerdata, simulatemardata, simstats, conditionvalue
include("./SimFunctions/simfunctions.jl")

end
