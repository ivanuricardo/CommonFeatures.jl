using Base: start_base_include

@testset "Mecm objective" begin
    using TensorToolbox, LinearAlgebra, Random, CommonFeatures, Zygote
    Random.seed!(20231228)

    N = [4, 3]
    mardata = randn(N[1], N[2], 100)
    ranks = [1, 2]
    maxiter = 100
    eta = 1e-04

    D = randn(N[1] * N[2])
    ΔY = randn(N[1] * N[2], 100)
    Y = cumsum(ΔY, dims=2)

    U1 = randn(N[1], ranks[1])
    U2 = randn(N[2], ranks[2])
    U3 = randn(N[1], ranks[1])
    U4 = randn(N[2], ranks[2])
    ϕ1 = randn(N[1], N[1])
    ϕ2 = randn(N[2], N[2])

    Σ1 = ϕ1 * ϕ1'
    Σ2 = ϕ2 * ϕ2'

end

using LinearAlgebra, Random, TensorToolbox, Plots, CommonFeatures, Zygote, ProgressBars
Random.seed!(20240920)

n = [4, 3]
ranks = [1, 1]
eta = 1e-05
p = 0
maxiter = 100

trueU1 = fill(NaN, n[1], ranks[1])
trueU2 = fill(NaN, n[2], ranks[2])
trueU3 = fill(NaN, n[1], ranks[1])
trueU4 = fill(NaN, n[2], ranks[2])
ct = 0

for i in 1:1000
    ct += 1

    U1, U2, U3, U4, ϕ1, ϕ2 = generatemecmparams(n, ranks, genphi=false)

    # Check I(1)
    i1cond = mecmstability(U1, U2, U3, U4, ϕ1, ϕ2)
    if i1cond
        trueU1 = U1
        trueU2 = U2
        trueU3 = U3
        trueU4 = U4
        println("I(1) condition satisfied")
        break
    end
end

obs = 1000
burnin = 100

mardata, flaty, lltrue = generatemecmdata(U1, U2, U3, U4, ϕ1, ϕ2, obs)

results = mecm(mardata, [1, 1]; p=0, eta=1e-05, maxiter=500, ϵ=1e-03, initcount=1, warmstart=1)
results.llist[1:findlast(!isnan, results.llist)]
startidx = 1
plot(results.llist[startidx:50])
plot(results.fullgrads)

grid = collect(Iterators.product(1:n[1], 1:n[2]))
ictable = fill(NaN, 5, prod(n))

for i in ProgressBar(1:prod(n))
    selectedrank = collect(grid[i])
    numpars = cointpar(n, ranks)
    mecmest = mecm(mardata, selectedrank; p=0, eta=1e-06, maxiter=200, ϵ=0.1)
    loglike = -mecmest.llist[findlast(!isnan, mecmest.llist)]
    ictable[1, i] = aic(loglike, numpars, obs)
    ictable[2, i] = bic(loglike, numpars, obs)
    ictable[3, i] = hqc(loglike, numpars, obs)
    ictable[4, i] = selectedrank[1]
    ictable[5, i] = selectedrank[2]
end

nancols = findall(x -> any(isnan, x), eachcol(ictable))
filteredic = ictable[:, setdiff(1:size(ictable, 2), nancols)]
aicvec = argmin(filteredic[1, :])
aicselected = Int.(filteredic[4:end, aicvec])
bicvec = argmin(filteredic[2, :])
bicselected = Int.(filteredic[4:end, bicvec])
hqcvec = argmin(filteredic[3, :])
hqcselected = Int.(filteredic[4:end, bicvec])
