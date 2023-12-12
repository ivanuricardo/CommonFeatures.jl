using TensorEconometrics, TensorToolbox, LinearAlgebra, Statistics

dimvals = [4, 3]
ranks = [2, 2, 2, 2]
obs = 500
scale = 2

r̄ = [4, 3, 4, 3]

simdata = simulatetuckerdata(dimvals, ranks, obs, scale)
mardata = simdata.tuckerdata
origy, lagy = tlag(mardata, 1)
N1, N2, obs = size(origy)

grid = collect(Iterators.product(1:4, 1:3, 1:4, 1:3))
myvec = collect(grid[6])

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

