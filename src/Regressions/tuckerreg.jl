
function regularize(a, b, U)
    norm_val = norm(U'U - b^2 * I)
    return a / 2 * norm_val^2
end

function objtuckreg(Yt, Xt, G, U1, U2, U3, U4, a, b)
    p = size(Xt, 3)
    obs = size(Xt, 4)

    A = kron(U2, U1) * tenmat(G, row=[1, 2]) * kron(I(p), kron(U4, U3))'

    eq = tenmat(Yt, col=3) - A * tenmat(Xt, col=4)

    obj = (1 / (2 * obs)) * norm(eq)^2

    facmats = [U1, U2, U3, U4]
    for i in facmats
        obj += regularize(a, b, i)
    end

    return obj
end

function dlbarest(origy, lagy, G, U1, U2, U3, U4, U5)
    A = full(ttensor(G, [U1, U2, U3, U4, U5]))
    N1, N2, p, obs = size(lagy)
    dlbar = zeros(N1, N2, N1, N2, p)
    innert = zeros(N1, N2)
    for i in 1:(obs)
        innert = contract(A, [3, 4, 5], lagy[:, :, :, i], [1, 2, 3])
        dlbar += ttt((innert - origy[:, :, i]), lagy[:, :, :, i])
    end
    dlbar .= dlbar ./ (obs)
end

"""
    tuckerreg(mardata, ranks::AbstractVector, eta=1e-04, a=1, b=1, ϵ=1e-04, maxiter=1000, p=1,ϵ=1e-02)

Perform a matrix autoregression with one lag (MAR(1)) with a reduced tucker rank along the coefficients.
Uses the gradient descent algorithm of Wang, Zhang, and Li 2024.

# Arguments
- `mardata::AbstractArray`: Input tensor data. Should be ``N1 \times N2 \times T``
- `ranks::AbstractVector`: Vector specifying the desired ranks of the Tucker decomposition.
- `eta::AbstractFloat`: Learning rate for gradient descent (default: 1e-04).
- `a::Real`: Regularization parameter (default: 1).
- `b::Real`: Regularization parameter (default: 1).
- `maxiter::Int`: Maximum number of iterations for gradient descent (default: 3000).
- `p::Int`: Number of lags to include
- `ϵ::AbstractFloat`: Convergence threshold for stopping criteria (default: 1e-04).

# Returns
A tuple containing the Tucker decomposition components:
- `G`: Core tensor.
- `U1`, `U2`, `U3`, `U4`: Factor matrices.
- `A`: Full Tucker decomposition tensor.
- `iters`: Number of iterations performed.
- `fullgrads`: A matrix keeping track of gradients. Can be plotted to determine whether gradients behave properly.
"""
using TensorToolbox, Statistics, Random, LinearAlgebra, TensorEconometrics, ProgressBars

Random.seed!(20231213)

sims = 100
dimvals = [4, 3]
ranks = [1, 1, 3, 3]
scale = 5
maxiter = 1000
eta = 1e-04
ϵ = 1e-03
a = 1
b = 1
maxeigen = 0.9
p = 2
obs = 500
medmar = simulatetuckerdata(dimvals, ranks, obs, scale, p, maxeigen)
mardata = medmar.data
function tuckerreg(mardata::AbstractArray, ranks::AbstractVector, eta::AbstractFloat=1e-04, a::Real=1, b::Real=1, maxiter::Int=1000, p::Int=1, ϵ::AbstractFloat=1e-02)
    N1, N2, _ = size(mardata)
    origy, lagy = tlag(mardata, p, true)

    initest = reshape(art(mardata, p), (N1, N2, N1, N2, p))
    ranks = vcat(ranks, p)
    hosvdinit = idhosvd(initest; reqrank=ranks)
    A = full(hosvdinit)

    U1, U2, U3, U4, U5 = hosvdinit.fmat
    G = ttm(hosvdinit.cten, U5', 5)
    U5 = U5' * U5

    trackG = fill(NaN, maxiter)
    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)

    iters = 0
    for s in 1:maxiter
        iters += 1

        dlbar1 = dlbarest(origy, lagy, G, U1, U2, U3, U4, U5)
        kronU1 = kron(U5, kron(U4, kron(U3, U2))) * tenmat(G, row=1)'
        regularizeU1 = a * (U1 * (U1'U1 - (b^2 * I)))
        ∇U1 = tenmat(dlbar1, row=1) * kronU1
        U1 -= eta * ∇U1 - eta * regularizeU1
        trackU1[s] = norm(∇U1)

        dlbar2 = dlbarest(origy, lagy, G, U1, U2, U3, U4, U5)
        kronU2 = kron(U5, kron(U4, kron(U3, U1))) * tenmat(G, row=2)'
        regularizeU2 = a * (U2 * (U2'U2 - (b^2 * I)))
        ∇U2 = tenmat(dlbar2, row=2) * kronU2
        U2 -= eta * ∇U2 - eta * regularizeU2
        trackU2[s] = norm(∇U2)

        dlbar3 = dlbarest(origy, lagy, G, U1, U2, U3, U4, U5)
        kronU3 = kron(U5, kron(U4, kron(U2, U1))) * tenmat(G, row=3)'
        regularizeU3 = a * (U3 * (U3'U3 - (b^2 * I)))
        ∇U3 = tenmat(dlbar3, row=3) * kronU3
        U3 -= eta * ∇U3 - eta * regularizeU3
        trackU3[s] = norm(∇U3)

        dlbar4 = dlbarest(origy, lagy, G, U1, U2, U3, U4, U5)
        kronU4 = kron(U5, kron(U3, kron(U2, U1))) * tenmat(G, row=4)'
        regularizeU4 = a * (U4 * (U4'U4 - (b^2 * I)))
        ∇U4 = tenmat(dlbar4, row=4) * kronU4
        U4 -= eta * ∇U4 - eta * regularizeU4
        trackU4[s] = norm(∇U4)

        dlbarG = dlbarest(origy, lagy, G, U1, U2, U3, U4, U5)
        facmat = [Matrix(U1'), Matrix(U2'), Matrix(U3'), Matrix(U4'), Matrix(U5)]
        ∇G = full(ttensor(dlbarG, facmat))
        trackG[s] = norm(∇G)
        G -= eta * ∇G

        A = full(ttensor(G, [U1, U2, U3, U4, Matrix(U5)]))

        # Stopping Condition
        c = trackU1[s] < ϵ && trackU2[s] < ϵ && trackU3[s] < ϵ && trackU4[s] < ϵ
        if c || s == maxiter
            fullgrads = hcat(trackU1, trackU2, trackU3, trackU4, trackG)
            A = idhosvd(A; reqrank=ranks)

            return (G=A.cten, U1=A.fmat[1], U2=A.fmat[2], U3=A.fmat[3],
                U4=A.fmat[4], A=full(A), iters=s, fullgrads=fullgrads)
        end
    end
end

function tuckerreg2(mardata::AbstractArray, ranks::AbstractVector, eta::AbstractFloat=1e-04, a::Real=1, b::Real=1, maxiter::Int=1000, p::Int=1, ϵ::AbstractFloat=1e-02)
    N1 = size(mardata, 1)
    N2 = size(mardata, 2)
    origy, lagy = tlag(mardata, p, true)

    initest = reshape(art(mardata, p), (N1, N2, N1, N2, p))
    ranks = vcat(ranks, p)
    hosvdinit = hosvd(initest; reqrank=ranks)
    A = full(hosvdinit)

    U1, U2, U3, U4, U5 = hosvdinit.fmat
    G = ttm(hosvdinit.cten, U5', 5)
    U5 = U5' * U5

    trackG = fill(NaN, maxiter)
    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)

    iters = 0
    for s in ProgressBar(1:maxiter)
        iters += 1

        ∇U1 = ReverseDiff.gradient(x -> objtuckreg(origy, lagy, G, x, U2, U3, U4, a, b), U1)
        U1 -= eta * ∇U1
        trackU1[s] = norm(∇U1)

        ∇U2 = ReverseDiff.gradient(x -> objtuckreg(origy, lagy, G, U1, x, U3, U4, a, b), U2)
        U2 -= eta * ∇U2
        trackU2[s] = norm(∇U2)

        ∇U3 = ReverseDiff.gradient(x -> objtuckreg(origy, lagy, G, U1, U2, x, U4, a, b), U3)
        U3 -= eta * ∇U3
        trackU3[s] = norm(∇U3)

        ∇U4 = ReverseDiff.gradient(x -> objtuckreg(origy, lagy, G, U1, U2, U3, x, a, b), U4)
        U4 -= eta * ∇U4
        trackU4[s] = norm(∇U4)

        ∇G = ReverseDiff.gradient(x -> objtuckreg(origy, lagy, x, U1, U2, U3, U4, a, b), G)
        trackG[s] = norm(∇G)
        G -= eta * ∇G

        A = full(ttensor(G, [U1, U2, U3, U4, Matrix(U5)]))

        # Stopping Condition
        c = trackU1[s] < ϵ && trackU2[s] < ϵ && trackU3[s] < ϵ && trackU4[s] < ϵ
        if c || s == maxiter
            fullgrads = hcat(trackU1, trackU2, trackU3, trackU4, trackG)
            A = hosvd(A; reqrank=ranks)
            U1, U2, U3, U4, U5 = A.fmat
            G = ttm(hosvdinit.cten, U5', 5)
            U5 = U5'U5
            Arot = full(ttensor(G, [U1, U2, U3, U4, U5]))
            return (G=G, U1=U1, U2=U2, U3=U3, U4=U4, A=Arot, iters=s, fullgrads=fullgrads)
        end
    end
end

