
function objmecm(ΔY, Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)
    _, obs = size(Y)
    sigma = -(obs / 2) * logdet(Σ1) - (obs / 2) * logdet(Σ2)
    U2U1 = kron(U2, U1)
    U4U3 = kron(U4, U3)'
    phi12 = kron(ϕ2, ϕ1)
    iΣ21 = inv(kron(Σ2, Σ1))
    ssr = 0
    for i in 2:obs
        res = ΔY[:, i] - U2U1 * U4U3 * Y[:, i] - phi12 * ΔY[:, i-1] - vec(D)
        ssr += res' * iΣ21 * res
    end
    return sigma - 0.5 * ssr
end

using LinearAlgebra, Random, TensorToolbox, Plots, CommonFeatures, Zygote
using Flux
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

function loss()
    return -objmecm(ΔY, Y, D, U1, U2, U3, U4, I(N1), I(N2), zeros(N1, N1), zeros(N2, N2))
end

opt = Flux.ADAM(1e-03)

Uhat1 = rand(3, 3)
Uhat2 = rand(3, 3)
Uhat3 = rand(3, 3)
Uhat4 = rand(3, 3)
Dhat = rand(3)
Σhat1 = Matrix{Float64}(I, 3, 3)
Σhat2 = Matrix{Float64}(I, 3, 3)
ϕhat1 = rand(3, 3)
ϕhat2 = rand(3, 3)

params = Flux.params(Uhat1, Uhat2, Uhat3, Uhat4, Dhat, Σhat1, Σhat2, ϕhat1, ϕhat2)

for epoch in 1:1000  # Adjust number of epochs
    Flux.train!(loss, params, opt)
    if epoch % 100 == 0
        println("Epoch: $epoch, Loss: $(loss())")
    end
end
