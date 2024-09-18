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
Random.seed!(20230915)

n = [4, 3]
ranks = [1, 3]
eta = 1e-05
p = 0
batchsize = 100
clipthresh = 20
maxiter = 100

u1 = 0.5 .* randn(n[1], ranks[1])
u2 = 0.5 .* randn(n[2], ranks[2])
u3 = 0.5 .* randn(n[1], ranks[1])
u4 = 0.5 .* randn(n[2], ranks[2])

# ϕ1 = 0.4 .* randn(n[1], n[1])
# ϕ2 = 0.4 .* randn(n[2], n[2])
# drift = 0.0001

kron21 = kron(u2, u1)
kron43 = kron(u4, u3)

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

results = mecm(mardata, [1, 3]; p=0, eta=1e-07, maxiter=400, ϵ=0.01, batchsize=100, clipthresh=75)
results.llist[1:findlast(!isnan, results.llist)]
startidx = 1
plot(results.llist[startidx:findlast(!isnan, results.llist)])
plot(results.fullgrads)

grid = collect(Iterators.product(1:n[1], 1:n[2]))
ictable = fill(NaN, 5, prod(n))

for i in ProgressBar(1:prod(n))
    selectedrank = collect(grid[i])
    numpars = cointpar(n, ranks)
    mecmest = mecm(mardata, selectedrank; p=0, eta=1e-04, maxiter=150, ϵ=0.01, clipthresh=50)
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
