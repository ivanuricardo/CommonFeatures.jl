module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra
using ProgressBars
using ReverseDiff

export tensorols, art
export tlag, ridgerankselect, rescaleten, idhosvd, spectralradius
export makecompanion, isstable
export infocrit, tuckerpar, fullinfocrit
export tuckerreg, tuckerreg2
export rrmarcrossval
include("./Regressions/MatrixAR.jl")
include("./Regressions/utils.jl")
include("./Regressions/companionmatrix.jl")
include("./Regressions/infocrit.jl")
include("./Regressions/tuckerreg.jl")
include("./Regressions/crossval.jl")

export simulatetuckerdata, simulatemardata, simstats, conditionvalue, generatetuckercoef
include("./SimFunctions/simfunctions.jl")

end
