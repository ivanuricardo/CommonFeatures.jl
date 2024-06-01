module CommonFeatures

using Statistics
using TensorToolbox
using LinearAlgebra
using ProgressBars
using ReverseDiff
using Distributions
using Combinatorics
using Parameters

import TensorToolbox.ttensor

export ReducedRankAutoRegression, LowRankTensorAutoRegression
export tensorols, art, rrvar
export tlag, ridgerankselect, rescaleten, idhosvd, spectralradius, ρ, vlag, ttensor
export makecompanion, isstable
export infocrit, tuckerpar, fullinfocrit, rrvaric
export tuckerreg, tuckerreg2, dlbarest
export rrmarcrossval
include("./Regressions/abstract.jl")
include("./Regressions/matrixar.jl")
include("./Regressions/regutils.jl")
include("./Regressions/companionmatrix.jl")
include("./Regressions/infocrit.jl")
include("./Regressions/tuckerreg.jl")
include("./Regressions/crossval.jl")

export simulatetuckerdata, generatetuckercoef
export simulatevardata, generatevarcoef, generaterrvarcoef, simulaterrvardata
export simstats
include("./SimFunctions/simtucker.jl")
include("./SimFunctions/simmar.jl")
include("./SimFunctions/simutils.jl")

end
