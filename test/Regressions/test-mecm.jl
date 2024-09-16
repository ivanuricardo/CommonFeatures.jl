
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

using LinearAlgebra, Random, TensorToolbox, PlotlyJS, CommonFeatures, Zygote, ProgressBars
Random.seed!(20230912)

n = [4, 3]
ranks = [1, 2]
eta = 1e-05
p = 0
batchsize = 100
maxiter = 100

u1 = 0.6 .* randn(n[1], ranks[1])
u2 = 0.6 .* randn(n[2], ranks[2])
u3 = 0.6 .* randn(n[1], ranks[1])
u4 = 0.6 .* randn(n[2], ranks[2])

# ϕ1 = 0.4 .* randn(n[1], n[1])
# ϕ2 = 0.4 .* randn(n[2], n[2])
drift = 0.0001

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

results = mecm(mardata, [1, 1]; p=0, eta=1e-05, maxiter=100, ϵ=0.1, batchsize=100)
results.llist
plot(results.fullgrads[15:end, :])
plot(results.llist)

results.llist[.!isnan.(results.llist)]


