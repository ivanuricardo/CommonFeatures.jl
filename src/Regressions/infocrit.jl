
"""
    tuckerpar(dimvals::AbstractVector, ranks::AbstractVector, P::Integer=1)

Compute the Tucker compression parameter for a tensor with specified dimensions and Tucker ranks.

# Arguments
- `dimvals::AbstractVector`: A vector representing the dimensions of the original tensor.
- `ranks::AbstractVector`: A vector representing the Tucker ranks for compression.
- `P::Integer=1`: An optional parameter representing the mode-n unfolding size (default is 1).

# Output
- Returns the Tucker compression parameter for the given input.

# Examples
```julia
dimvals = [3, 4, 5]
ranks = [2, 3, 2, 4]
P = 2
result = tuckerpar(dimvals, ranks, P)
println(result)
```
# References
- Tucker, L. R. (1966). Some mathematical notes on three-mode factor analysis. Psychometrika, 31(3), 279-311.
"""
function tuckerpar(dimvals::AbstractVector, ranks::AbstractVector, P::Integer=1)
    k = length(ranks) ÷ 2
    totalsum = prod(ranks)
    for i in 1:k
        totalsum += ranks[i] * (dimvals[i] - ranks[i])
    end
    for i in 1:(k-1)
        totalsum += ranks[k+i] * (dimvals[i] - ranks[k+i])
    end
    totalsum += ranks[2*k] * (dimvals[k] * P - ranks[2*k])
    return totalsum
end

"""
    infocrit(mardata, p, r̄)

Calculate information criteria (AIC and BIC) for different combinations of Tucker ranks in a matrix autoregressive (MAR) model.

# Arguments
- `mardata::AbstractArray`: The matrix-valued time series data.
- `p::Int`: The order of the autoregressive model.
- `r̄::AbstractVector`: A vector specifying the maximum Tucker ranks for each mode. Default is all possible combinations.
- `tuckiter::Int`: An integer specifying the number of iterations the Tucker regression should run. Default value is 500.
- `tucketa::Real`: A real value specifying the step size for the Tucker regression. Default value is 1e-05

# Output
A tuple with the following elements:
- `BIC`: The Tucker ranks chosen based on the Bayesian Information Criterion (BIC).
- `AIC`: The Tucker ranks chosen based on the Akaike Information Criterion (AIC).
- `ictable`: A 6xN matrix where N is the number of valid Tucker rank combinations. Each column corresponds to a combination, and rows contain the following information:
  - Row 1: Log determinant of the covariance matrix plus a penalty term (BIC criterion).
  - Row 2: Log determinant of the covariance matrix plus a penalty term (AIC criterion).
  - Rows 3-6: The chosen Tucker ranks for each mode.

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
function infocrit(mardata::AbstractArray, p::Int, r̄::AbstractVector=[], a::Real=1, b::Real=1, tuckiter::Int=500, tucketa::Real=1e-05, fixedeta::Bool=true, orthonorm::Bool=true)
    # Each row is associated with either AIC, BIC, and the assocaited rank
    origy, lagy = tlag(mardata, p)
    N1, N2, obs = size(origy)
    if r̄ == []
        r̄ = [N1, N2, N1, N2]
    end
    infocritest = fill(NaN, 6, prod(r̄))
    regiters = fill(NaN, prod(r̄))
    grid = collect(Iterators.product(1:r̄[1], 1:r̄[2], 1:r̄[3], 1:r̄[4]))
    for i in 1:prod(r̄)
        selectedrank = collect(grid[i])
        r1, r2, r3, r4 = selectedrank
        if r1 > r2 * r3 * r4 || r2 > r1 * r3 * r4 || r3 > r1 * r2 * r4 || r4 > r1 * r2 * r3
            infocritest[3, i] = r1
            infocritest[4, i] = r2
            infocritest[5, i] = r3
            infocritest[6, i] = r4
        else
            # tuckest = naivetuckreg(mardata, [i, j, k, l], p)
            tuckest = tuckerreg(mardata, selectedrank, tucketa, a, b, tuckiter, 1, fixedeta, orthonorm)
            err = origy - contract(tuckest.A, [3, 4], lagy, [1, 2])
            flatϵ = tenmat(err, col=3)
            detcov = det(flatϵ * flatϵ')
            infocritest[1, i] = log(detcov) + (2 * tuckerpar([N1, N2], selectedrank)) / obs
            infocritest[2, i] = log(detcov) + (tuckerpar([N1, N2], selectedrank) * log(obs)) / obs
            infocritest[3, i] = r1
            infocritest[4, i] = r2
            infocritest[5, i] = r3
            infocritest[6, i] = r4
            regiters[i] = tuckest.iters
        end
    end
    nancols = findall(x -> any(isnan, x), eachcol(infocritest))
    filteredic = infocritest[:, setdiff(1:size(infocritest, 2), nancols)]
    AICvec = argmin(filteredic[1, :])
    AICchosen = Int.(filteredic[3:end, AICvec])
    BICvec = argmin(filteredic[2, :])
    BICchosen = Int.(filteredic[3:end, BICvec])
    truetuck = tuckerreg(mardata, BICchosen, tucketa, a, b, tuckiter, 1, fixedeta, orthonorm)

    return (BIC=BICchosen, AIC=AICchosen, ictable=infocritest, regiters=regiters, truetuck=truetuck)
end
