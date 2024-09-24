using LinearAlgebra, Random, TensorToolbox, Plots, CommonFeatures, Zygote, ProgressBars
Random.seed!(20240922)

n = [4, 3]
ranks = [4, 1]
eta = 2e-07
p = 0
maxiter = 100

trueU1 = fill(NaN, n[1], ranks[1])
trueU2 = fill(NaN, n[2], ranks[2])
trueU3 = fill(NaN, n[1], ranks[1])
trueU4 = fill(NaN, n[2], ranks[2])
trueϕ1 = zeros(n[1], n[1])
trueϕ2 = zeros(n[2], n[2])
ct = 0

for i in 1:1000
    ct += 1

    U1, U2, U3, U4, ϕ1, ϕ2 = generatemecmparams(n, ranks, genphi=false)

    # Check I(1)
    i1cond = mecmstable(U1, U2, U3, U4, ϕ1, ϕ2)
    if maximum(i1cond) < 0.9
        trueU1 = U1
        trueU2 = U2
        trueU3 = U3
        trueU4 = U4
        println("I(1) condition satisfied")
        break
    end
end
mecmstable(trueU1, trueU2, trueU3, trueU4, trueϕ1, trueϕ2)

obs = 1000
burnin = 100

mardata, flaty, lltrue = generatemecmdata(trueU1, trueU2, trueU3, trueU4, trueϕ1, trueϕ2, obs)

results = mecm(mardata, [4, 1]; p=0, maxiter=50, ϵ=1e-02)
results.llist[1:findlast(!isnan, results.llist)]
startidx = 20
plot(results.llist[startidx:findlast(!isnan, results.llist)])
plot(results.fullgrads)

grid = collect(Iterators.product(1:n[1], 1:n[2]))
ictable = fill(NaN, 5, prod(n))

for i in ProgressBar(1:prod(n))
    selectedrank = collect(grid[i])
    numpars = cointpar(n, ranks)
    mecmest = mecm(mardata, selectedrank; p=0, maxiter=50, ϵ=1e-03)
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
