module CommonFeatures

using Statistics
using TensorToolbox
using LinearAlgebra
using ProgressBars
using ReverseDiff
using Zygote
using Distributions
using Combinatorics
using Parameters

import TensorToolbox.ttensor

export ReducedRankAutoRegression, LowRankTensorAutoRegression
export tensorols, art, rrvar
export tlag, ridgerankselect, rescaleten, idhosvd, spectralradius, ρ, vlag, ttensor
export makecompanion, isstable
export infocrit, tuckerpar, fullinfocrit, rrvaric, aic, bic, hqc, cointpar
export tuckerreg, tuckerreg2, dlbarest
export mecm, mecm2, objmecm, matobj, mecminit
export rrmarcrossval
export mecmsumres, U1grad, U1hessian, U2grad, U2hessian, U3grad, U3hessian
export U4grad, U4hessian, ϕ1grad, ϕ1hessian, ϕ2grad, ϕ2hessian, Σ1grad, Σ2grad
include("./Regressions/abstract.jl")
include("./Regressions/matrixar.jl")
include("./Regressions/regutils.jl")
include("./Regressions/companionmatrix.jl")
include("./Regressions/infocrit.jl")
include("./Regressions/tuckerreg.jl")
include("./Regressions/mecm.jl")
include("./Regressions/crossval.jl")
include("./Regressions/mecmgrads.jl")

export simulatetuckerdata, generatetuckercoef
export simulatevardata, generatevarcoef, generaterrvarcoef, simulaterrvardata
export simstats
export rorth, mecmstable, generatemecmparams, generatemecmdata, selectmecm
include("./SimFunctions/simtucker.jl")
include("./SimFunctions/simmar.jl")
include("./SimFunctions/simutils.jl")
include("./SimFunctions/simmecm.jl")

end
