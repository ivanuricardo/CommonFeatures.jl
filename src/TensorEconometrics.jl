module TensorEconometrics

using Statistics
using TensorToolbox
using LinearAlgebra
using ProgressBars
using ReverseDiff
using Distributions

export tensorols, art, rrvar
export tlag, ridgerankselect, rescaleten, idhosvd, spectralradius, vlag
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
