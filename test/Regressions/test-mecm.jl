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
ranks = [1, 3]
eta = 1e-05
p = 0
batchsize = 100
clipthresh = 20
maxiter = 100

trueU1 = fill(NaN, n[1], ranks[1])
trueU2 = fill(NaN, n[2], ranks[2])
trueU3 = fill(NaN, n[1], ranks[1])
trueU4 = fill(NaN, n[2], ranks[2])
for i in 1:1000
    Q1 = rorth(n[1], ranks[1])
    Q2 = rorth(n[1], ranks[1])
    Q3 = rorth(n[2], ranks[2])
    Q4 = rorth(n[2], ranks[2])

    A = Q1 * diagm(randn(ranks[1])) * Q2'
    B = Q3 * diagm(randn(ranks[2])) * Q4'

    u2, s1, u4 = svd(A)
    u1, s2, u3 = svd(B)
    trueU1 .= 1.5 .* u1[:, 1:ranks[1]] * diagm(s2[1:ranks[1]])
    trueU2 .= 1.5 .* u2[:, 1:ranks[2]] * diagm(s1[1:ranks[2]])
    trueU3 .= 1.5 .* u3[:, 1:ranks[1]]
    trueU4 .= 1.5 .* u4[:, 1:ranks[2]]

    # ϕ1 = 0.4 .* randn(n[1], n[1])
    # ϕ2 = 0.4 .* randn(n[2], n[2])
    # drift = 0.0001

    kron21 = kron(trueU2, trueU1)
    kron43 = kron(trueU4, trueU3)

    # Check I(1)
    i1cond = (abs.(eigvals(I + kron43'kron21))[1] < 0.9)
    if i1cond
        println("I(1) condition satisfied")
        break
    end
end

obs = 1000
burnin = 100

Y = zeros(n[1] * n[2], obs + burnin)
for i in 3:(obs+burnin)
    piy = kron21 * kron43' * Y[:, i-1]
    # phideltay = kron(ϕ2, ϕ1) * (Y[:, i-1] - Y[:, i-2])
    Y[:, i] .= Y[:, i-1] + piy + randn(n[1] * n[2])
end
mardata = matten(Y[:, (burnin+1):end], [1, 2], [3], [n[1], n[2], obs])

mdy = mardata[:, :, 2:end] - mardata[:, :, 1:end-1]
my = mardata[:, :, 1:end-1]
N1, N2, obs = size(my)
ΔY = reshape(mdy, N1 * N2, obs)
Y = reshape(my, N1 * N2, obs)
D = zeros(size(mdy, 1) * size(mdy, 2))
lltrue = objmecm(ΔY, Y, D, u1, u2, u3, u4, I(N1), I(N2), zeros(N1, N1), zeros(N2, N2))

results = mecm(mardata, [1, 3]; p=0, eta=1e-05, maxiter=100, ϵ=1.0, batchsize=100, clipthresh=10000)
results.llist[1:findlast(!isnan, results.llist)]
startidx = 5
plot(results.llist[startidx:findlast(!isnan, results.llist)])
plot(results.fullgrads)

grid = collect(Iterators.product(1:n[1], 1:n[2]))
ictable = fill(NaN, 5, prod(n))

for i in ProgressBar(1:prod(n))
    selectedrank = collect(grid[i])
    numpars = cointpar(n, ranks)
    mecmest = mecm(mardata, selectedrank; p=0, eta=1e-05, maxiter=100, ϵ=1.0, clipthresh=10000)
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
