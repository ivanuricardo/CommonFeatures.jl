module CommonFeatures

using Statistics
using TensorToolbox
using LinearAlgebra
using ProgressBars
using ReverseDiff
using Distributions

export tensorols, art, rrvar
export tlag, ridgerankselect, rescaleten, idhosvd, spectralradius, vlag
export makecompanion, isstable
export infocrit, tuckerpar, fullinfocrit, rrvaric
export tuckerreg, tuckerreg2, dlbarest
export rrmarcrossval
include("./Regressions/MatrixAR.jl")
include("./Regressions/utils.jl")
include("./Regressions/companionmatrix.jl")
include("./Regressions/infocrit.jl")
include("./Regressions/tuckerreg.jl")
include("./Regressions/crossval.jl")

export simulatetuckerdata, generatetuckercoef
export simulatemardata, generatemarcoef, simulatevardata, generatevarcoef, generaterrvarcoef, simulaterrvardata
export simstats
include("./SimFunctions/simtucker.jl")
include("./SimFunctions/simmar.jl")
include("./SimFunctions/utils.jl")

end
