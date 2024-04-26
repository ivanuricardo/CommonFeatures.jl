
"""
    tuckerpar(dimvals::AbstractVector, ranks::AbstractVector, P::Integer=1)

Compute the Tucker compression parameter for a tensor with specified dimensions and Tucker ranks.

# Arguments
- `dimvals::AbstractVector`: A vector representing the dimensions of the original tensor.
- `ranks::AbstractVector`: A vector representing the Tucker ranks for compression. It should have twice the length of `dimvals`.
- `P::Integer=1`: An optional parameter representing the mode-n unfolding size (default is 1).

# Output
- Returns the Tucker compression parameter for the given input.

# Examples
```julia
dimvals = [3, 4, 5]
ranks = [2, 3, 2, 4]
P = 2
result = tuckerpar(dimvals, ranks, P)
println(result)  # Output: 44
```
# References
- Tucker, L. R. (1966). Some mathematical notes on three-mode factor analysis. Psychometrika, 31(3), 279-311.
"""
function tuckerpar(dimvals::AbstractVector, ranks::AbstractVector, p::Integer=1)
    k = length(ranks) ÷ 2
    totalsum = prod(ranks) * p
    for i in 1:k
        term1 = ranks[i] * (dimvals[i] - ranks[i])
        term2 = ranks[k+i] * (dimvals[i] - ranks[k+i])
        totalsum += term1 + term2
    end
    return totalsum
end

function AIC(logdet::Real, numpars::Int, obs::Int)
    return logdet + (2 * numpars) / obs
end

function BIC(logdet::Real, numpars::Int, obs::Int)
    return logdet + (numpars * log(obs)) / obs
end
function HQ(logdet::Real, numpars::Int, obs::Int)
    return logdet + (numpars * 2 * log(log(obs))) / obs
end

function tuckercondition(r::Vector{Int})
    n = length(r)
    for i in 1:n
        prod_except_i = prod(r[j] for j in 1:n if j != i)
        if r[i] > prod_except_i
            return false
        end
    end
    return true
end

"""
    infocrit(mardata, p, r̄, maxiters, tucketa, ϵ, stdize)

Calculate information criteria (AIC and BIC) for different combinations of Tucker ranks in a matrix autoregressive (MAR) model.

# Arguments
- `mardata::AbstractArray`: The matrix-valued time series data.
- `p::Int`: The order of the autoregressive model.
- `r̄::AbstractVector`: A vector specifying the maximum Tucker ranks for each mode. Default is all possible combinations.
- `maxiters::Int`: An integer specifying the number of iterations the Tucker regression should run. Default value is 500.
- `tucketa::Real`: A real value specifying the step size for the Tucker regression. Default value is 1e-02.
- `ϵ::Real`: A real value specifying the convergence criterion for the Tucker regression. Default value is 1e-03.
- `stdize::Bool`: A boolean value specifying whether to standardize the data before running the Tucker regression. Default value is false.

# Output
A tuple with the following elements:
- `BIC`: The Tucker ranks chosen based on the Bayesian Information Criterion (BIC).
- `AIC`: The Tucker ranks chosen based on the Akaike Information Criterion (AIC).
- `ictable`: A 6xN matrix where N is the number of valid Tucker rank combinations. Each column corresponds to a combination, and rows contain the following information:
  - Row 1: Log determinant of the covariance matrix plus a penalty term (BIC criterion).
  - Row 2: Log determinant of the covariance matrix plus a penalty term (AIC criterion).
  - Rows 3-6: The chosen Tucker ranks for each mode.
  - Row 7: The number of iterations for each Tucker regression.
- numconv: The number of converged Tucker regressions.

# Example
```julia
mardata = randn(4,3,100)  # Example matrix time series data
p = 2
r̄ = [2, 2, 2, 2]
result = infocrit(mardata, p, r̄)
println("BIC Chosen Ranks: ", result.BIC)
println("AIC Chosen Ranks: ", result.AIC)
println("Information Criteria Table: ", result.ictable)
```
"""
function infocrit(
    mardata::AbstractArray,
    p::Int,
    r̄::AbstractVector=[],
    maxiters::Int=500,
    tucketa::Real=1e-02,
    ϵ::Real=1e-03,
    stdize::Bool=false
)
    origy, _ = tlag(mardata, p)
    N1, N2, obs = size(origy)
    if isempty(r̄)
        r̄ = [N1, N2, N1, N2]
    end

    infocritest = fill(NaN, 7, prod(r̄))
    grid = collect(Iterators.product(1:r̄[1], 1:r̄[2], 1:r̄[3], 1:r̄[4]))
    numconv = 0
    # Threads.@threads for i in ProgressBar(1:prod(r̄))
    for i in 1:prod(r̄)
        selectedrank = collect(grid[i])
        r1, r2, r3, r4 = selectedrank
        if tuckercondition(selectedrank)
            infocritest[3, i] = r1
            infocritest[4, i] = r2
            infocritest[5, i] = r3
            infocritest[6, i] = r4
            continue
        end
        numpars = tuckerpar([N1, N2], selectedrank, p)

        tuckest = tuckerreg(mardata, selectedrank, tucketa, maxiters, p, ϵ, stdize)
        tuckerr = tuckest.residuals
        logdetcov = logdet(tuckerr * tuckerr' / obs)

        if tuckest.converged == true
            numconv += 1
        end

        infocritest[1, i] = AIC(logdetcov, numpars, obs)
        infocritest[2, i] = BIC(logdetcov, numpars, obs)
        infocritest[3, i] = r1
        infocritest[4, i] = r2
        infocritest[5, i] = r3
        infocritest[6, i] = r4
        infocritest[7, i] = tuckest.iters
    end
    nancols = findall(x -> any(isnan, x), eachcol(infocritest))
    filteredic = infocritest[:, setdiff(1:size(infocritest, 2), nancols)]
    AICvec = argmin(filteredic[1, :])
    AICchosen = Int.(filteredic[3:end, AICvec])
    BICvec = argmin(filteredic[2, :])
    BICchosen = Int.(filteredic[3:end, BICvec])

    return (BIC=BICchosen, AIC=AICchosen, ictable=infocritest, numconv=numconv)
end

"""
    fullinfocrit(mardata, p, r̄, maxiters, tucketa, ϵ, stdize)

Calculate information criteria (AIC and BIC) for different combinations of Tucker ranks in a matrix autoregressive (MAR) model.
Iterates over different time lags to find the best Tucker ranks.

# Fields
- `mardata::AbstractArray`: The matrix-valued time series data.
- `pmax::Int`: The maximum order of the autoregressive model to be considered.
- `r̄::AbstractVector`: A vector specifying the maximum Tucker ranks for each mode. Default is all possible combinations.
- `maxiters::Int`: An integer specifying the number of iterations the Tucker regression should run. Default value is 500.
- `tucketa::Real`: A real value specifying the step size for the Tucker regression. Default value is 1e-02.
- `ϵ::Real`: A real value specifying the convergence criterion for the Tucker regression. Default value is 1e-03.
- `stdize::Bool`: A boolean value specifying whether to standardize the data before running the Tucker regression. Default value is false.

# Output
A tuple with the following elements:
- `BIC`: The Tucker ranks chosen based on the Bayesian Information Criterion (BIC).
- `AIC`: The Tucker ranks chosen based on the Akaike Information Criterion (AIC).
- `ictable`: A 6xN matrix where N is the number of valid Tucker rank combinations. Each column corresponds to a combination, and rows contain the following information:
  - Row 1: Log determinant of the covariance matrix plus a penalty term (BIC criterion).
  - Row 2: Log determinant of the covariance matrix plus a penalty term (AIC criterion).
  - Rows 3-6: The chosen Tucker ranks for each mode.
- regiters: The number of iterations for each Tucker regression.
- numconv: The number of converged Tucker regressions.

# Example
```julia
mardata = randn(4,3,100)  # Example matrix time series data
p = 2
r̄ = [2, 2, 2, 2]
result = infocrit(mardata, p, r̄)
println("BIC Chosen Ranks: ", result.BIC)
println("AIC Chosen Ranks: ", result.AIC)
println("Information Criteria Table: ", result.ictable)
```
"""
function fullinfocrit(mardata::AbstractArray, pmax::Int, r̄::AbstractVector=[], maxiters::Int=1000, tucketa::Real=1e-04, ϵ::Real=1e-01, stdize::Bool=false)
    origy, _ = tlag(mardata, pmax)
    N1, N2, obs = size(origy)
    if isempty(r̄)
        r̄ = [N1, N2, N1, N2]
    end

    infocritest = fill(NaN, 8, prod(r̄) * pmax)
    regiters = fill(NaN, prod(r̄) * pmax)
    grid = collect(Iterators.product(1:r̄[1], 1:r̄[2], 1:r̄[3], 1:r̄[4], 1:pmax))
    numconv = 0
    Threads.@threads for i in ProgressBar(1:(prod(r̄)*pmax))
        selectedrank = collect(grid[i])
        r1, r2, r3, r4, p = selectedrank
        if tuckercondition(selectedrank)
            infocritest[4, i] = r1
            infocritest[5, i] = r2
            infocritest[6, i] = r3
            infocritest[7, i] = r4
            infocritest[8, i] = p
            continue
        end
        if p == 1
            tuckest = tuckerreg(mardata[:, :, pmax:end], [r1, r2, r3, r4], tucketa, maxiters, p, ϵ, stdize)
        elseif p == 2
            tuckest = tuckerreg(mardata[:, :, (pmax-1):end], [r1, r2, r3, r4], tucketa, maxiters, p, ϵ, stdize)
        elseif p == 3
            tuckest = tuckerreg(mardata[:, :, (pmax-2):end], [r1, r2, r3, r4], tucketa, maxiters, p, ϵ, stdize)
        elseif p == 4
            tuckest = tuckerreg(mardata[:, :, (pmax-3):end], [r1, r2, r3, r4], tucketa, maxiters, p, ϵ, stdize)
        end
        tuckerr = tuckest.residuals
        logdetcov = logdet(tuckerr * tuckerr' / obs)
        numpars = tuckerpar([N1, N2], [r1, r2, r3, r4], p)

        if tuckest.converged == true
            numconv += 1
        end

        infocritest[1, i] = AIC(logdetcov, numpars, obs)
        infocritest[2, i] = BIC(logdetcov, numpars, obs)
        infocritest[3, i] = HQ(logdetcov, numpars, obs)
        infocritest[4, i] = r1
        infocritest[5, i] = r2
        infocritest[6, i] = r3
        infocritest[7, i] = r4
        infocritest[8, i] = p
        regiters[i] = tuckest.iters
    end
    nancols = findall(x -> any(isnan, x), eachcol(infocritest))
    filteredic = infocritest[:, setdiff(1:size(infocritest, 2), nancols)]
    AICvec = argmin(filteredic[1, :])
    AICchosen = Int.(filteredic[4:end, AICvec])
    BICvec = argmin(filteredic[2, :])
    BICchosen = Int.(filteredic[4:end, BICvec])
    HQvec = argmin(filteredic[3, :])
    HQchosen = Int.(filteredic[4:end, HQvec])

    return (BIC=BICchosen, AIC=AICchosen, HQ=HQchosen, ictable=infocritest, regiters=regiters, numconv=numconv)
end

"""
    rrvaric(vardata, pmax, stdize)

Compute the optimal rank of a reduced rank regression using information criteria.

# Fields
- `vardata::AbstractMatrix`: The data matrix where each column represents a variable and each row represents an observation.
- `pmax::Int`: The maximum lag order.
- `stdize::Bool`: Indicates whether to standardize the data or not.

# Returns
A tuple with the following elements:
- BIC: The optimal rank based on the Bayesian Information Criterion.
- AIC: The optimal rank based on the Akaike Information Criterion.
- HQ: The optimal rank based on the Hannan-Quinn Information Criterion.
- ictable: A 5xN matrix where N is the number of valid rank combinations. Each column corresponds to a combination, and rows contain the following information:
  - Row 1: Log determinant of the covariance matrix plus a penalty term (BIC criterion).
  - Row 2: Log determinant of the covariance matrix plus a penalty term (AIC criterion).
  - Row 3: Log determinant of the covariance matrix plus a penalty term (HQ criterion).
  - Row 4: The chosen rank.
  - Row 5: The chosen lag order.
"""
function rrvaric(vardata::AbstractMatrix, pmax::Int, stdize::Bool)
    k, obs = size(vardata)
    resp = vlag(vardata, pmax)[1:k, :]
    resp = resp .- mean(resp, dims=2)
    pred = vlag(vardata, pmax)[(k+1):end, :]
    pred = pred .- mean(pred, dims=2)

    infocritest = fill(NaN, 5, prod(k) * pmax)
    grid = collect(Iterators.product(1:k, 1:pmax))

    for i in 1:(prod(k)*pmax)
        selectedrank = collect(grid[i])
        r, p = selectedrank
        if p == 1
            rrvarest = rrvar(vardata[:, pmax:end], r, p, stdize)
        elseif p == 2
            rrvarest = rrvar(vardata[:, (pmax-1):end], r, p, stdize)
        elseif p == 3
            rrvarest = rrvar(vardata[:, (pmax-2):end], r, p, stdize)
        elseif p == 4
            rrvarest = rrvar(vardata[:, (pmax-3):end], r, p, stdize)
        end
        rrvarerr = rrvarest.rrvarerr
        logdetcov = logdet(rrvarerr * rrvarerr' / obs)
        numpars = (k * r) + (k * r * p)

        infocritest[1, i] = AIC(logdetcov, numpars, obs)
        infocritest[2, i] = BIC(logdetcov, numpars, obs)
        infocritest[3, i] = HQ(logdetcov, numpars, obs)
        infocritest[4, i] = r
        infocritest[5, i] = p
    end

    AICvec = argmin(infocritest[1, :])
    AICchosen = Int.(infocritest[4:end, AICvec])
    BICvec = argmin(infocritest[2, :])
    BICchosen = Int.(infocritest[4:end, BICvec])
    HQvec = argmin(infocritest[3, :])
    HQchosen = Int.(infocritest[4:end, HQvec])

    return (BIC=BICchosen, AIC=AICchosen, HQ=HQchosen, ictable=infocritest)

end
