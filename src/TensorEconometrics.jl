module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra
using ProgressBars

export tensorols, art
export tlag, ridgerankselect, rescaleten
export makecompanion, isstable
export infocrit, tuckerpar
export tuckerreg
export rrmarcrossval
include("./Regressions/MatrixAR.jl")
include("./Regressions/utils.jl")
include("./Regressions/companionmatrix.jl")
include("./Regressions/infocrit.jl")
include("./Regressions/tuckerreg.jl")
include("./Regressions/crossval.jl")

export simulatetuckerdata, simulatemardata, simstats, conditionvalue
include("./SimFunctions/simfunctions.jl")

end
