
"""
    infocrit(mardata, p, r̄)

Calculate information criteria (AIC and BIC) for different combinations of Tucker ranks in a matrix autoregressive (MAR) model.

# Arguments
- `mardata::AbstractArray`: The matrix-valued time series data.
- `p::Int`: The order of the autoregressive model.
- `r̄::AbstractVector`: A vector specifying the maximum Tucker ranks for each mode. Default is all possible combinations.

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
function infocrit(mardata::AbstractArray, p::Int; r̄::AbstractVector)
    initest = art(mardata, p)
    # Each row is associated with either AIC, BIC, and the assocaited rank
    infocritest = fill(NaN, 6, prod(r̄))
    origy, lagy = tlag(mardata, p)
    N1, N2, obs = size(origy)
    counter = 0
    for i in 1:r̄[1]
        for j in 1:r̄[2]
            for k in 1:r̄[3]
                for l in 1:r̄[4]
                    counter += 1

                    if i > j * k * l || j > i * k * l || k > i * j * l || l > i * j * k
                        infocritest[3, counter] = i
                        infocritest[4, counter] = j
                        infocritest[5, counter] = k
                        infocritest[6, counter] = l
                    else
                        tuckest = tuckerreg(mardata, [i, j, k, l], initest, 1e-05, 1, 1, 0.1, 1000)
                        ϵ = origy - contract(tuckest.A, [3, 4], lagy, [1, 2])
                        flatϵ = tenmat(ϵ, col=3)
                        detcov = det(flatϵ * flatϵ')
                        infocritest[1, counter] = log(detcov) + (2 * tuckerpar([N1, N2], [i, j, k, l])) / obs
                        infocritest[2, counter] = log(detcov) + (tuckerpar([N1, N2], [i, j, k, l]) * log(obs)) / obs
                        infocritest[3, counter] = i
                        infocritest[4, counter] = j
                        infocritest[5, counter] = k
                        infocritest[6, counter] = l
                    end
                end
            end
        end
    end
    nancols = findall(x -> any(isnan, x), eachcol(infocritest))
    filteredic = infocritest[:, setdiff(1:size(infocritest, 2), nancols)]
    AICvec = argmin(filteredic[1, :])
    AICchosen = filteredic[3:end, AICvec]
    BICvec = argmin(filteredic[2, :])
    BICchosen = filteredic[3:end, BICvec]

    return (BIC=BICchosen, AIC=AICchosen, ictable=infocritest)
end
