using LinearAlgebra, Random, TensorToolbox, Plots, CommonFeatures, Zygote, ProgressBars
Random.seed!(20240921)

N1 = 4
N2 = 3
r1 = 2
r2 = 1
obs = 100

U1, U2, U3, U4, ϕ1, ϕ2 = generatemecmparams([N1, N2], [r1, r2], genphi=true)
ΔY = randn(N1, N2, obs)
Y = randn(N1, N2, obs)
D = zeros(N1, N2)
Σ1, Σ2 = I(N1), I(N2)

ll = loss(ΔY, Y, D, U1, U2, U3, U4, ϕ1, ϕ2)
ll2 = objmecm(tenmat(ΔY, row=[1, 2]), tenmat(Y, row=[1, 2]), D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)

truegradD = gradient(x -> loss(ΔY, Y, x, U1, U2, U3, U4, ϕ1, ϕ2), D)[1]
truegradD + mecmsumres(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)

truegrad1 = gradient(x -> loss(ΔY, Y, D, x, U2, U3, U4, ϕ1, ϕ2), U1)[1]
truegrad1 - U1grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)

truegrad2 = gradient(x -> loss(ΔY, Y, D, U1, x, U3, U4, ϕ1, ϕ2), U2)[1]
truegrad2 - U2grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)

truegrad3 = gradient(x -> loss(ΔY, Y, D, U1, U2, x, U4, ϕ1, ϕ2), U3)[1]
truegrad3 - U3grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)

truegrad4 = gradient(x -> loss(ΔY, Y, D, U1, U2, U3, x, ϕ1, ϕ2), U4)[1]
truegrad4 - U4grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)

truegradϕ1 = gradient(x -> loss(ΔY, Y, D, U1, U2, U3, U4, x, ϕ2), ϕ1)[1]
truegradϕ1 - ϕ1grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)

truegradϕ2 = gradient(x -> loss(ΔY, Y, D, U1, U2, U3, U4, ϕ1, x), ϕ2)[1]
truegradϕ2 - ϕ2grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)

1 / maximum(eigvals(hessian(x -> loss(ΔY, Y, D, x, U2, U3, U4, ϕ1, ϕ2), U1)))
1 / maximum(eigvals(U1hessian(Y, U2, U3, U4)))

1 / maximum(eigvals(hessian(x -> loss(ΔY, Y, D, U1, x, U3, U4, ϕ1, ϕ2), U2)))
1 / maximum(eigvals(U2hessian(Y, U1, U3, U4)))

1 / maximum(eigvals(hessian(x -> loss(ΔY, Y, D, U1, U2, x, U4, ϕ1, ϕ2), U3)))
1 / maximum(eigvals(U3hessian(Y, U1, U2, U4)))

1 / maximum(eigvals(hessian(x -> loss(ΔY, Y, D, U1, U2, U3, x, ϕ1, ϕ2), U4)))
1 / maximum(eigvals(U4hessian(Y, U1, U2, U3)))

1 / maximum(eigvals(hessian(x -> loss(ΔY, Y, D, U1, U2, U3, U4, x, ϕ2), ϕ1)))
1 / maximum(eigvals(ϕ1hessian(ΔY, ϕ2)))

1 / maximum(eigvals(hessian(x -> loss(ΔY, Y, D, U1, U2, U3, U4, ϕ1, x), ϕ2)))
1 / maximum(eigvals(ϕ2hessian(ΔY, ϕ1)))

