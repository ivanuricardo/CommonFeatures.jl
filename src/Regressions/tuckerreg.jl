
function objtuckreg(Yt, Xt, G, U1, U2, U3, U4)
    p = size(Xt, 3)
    obs = size(Xt, 4)

    A = kron(U2, U1) * tenmat(G, row=[1, 2]) * kron(I(p), kron(U4, U3))'

    eq = tenmat(Yt, col=3) - A * tenmat(Xt, col=4)

    return (1 / (2 * obs)) * norm(eq)^2
end

function dlbarest(origy, lagy, G, U1, U2, U3, U4, U5)
    A = full(ttensor(G, [U1, U2, U3, U4, U5]))
    N1, N2, p, obs = size(lagy)
    dlbar = zeros(N1, N2, N1, N2, p)
    innert = zeros(N1, N2)
    for i in 1:(obs)
        tena = reshape(A, (N1 * N2, N1 * N2 * p))
        innert = reshape(tena * vec(lagy[:, :, :, i]), (N1, N2))
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
function tuckerreg(mardata::AbstractArray, ranks::AbstractVector, eta::AbstractFloat=1e-04, maxiter::Int=1000, p::Int=1, ϵ::AbstractFloat=1e-02)
    origy, lagy = tlag(mardata, p)
    N1, N2, _ = size(origy)
    cenorig = origy .- mean(origy, dims=3)
    cenlag = lagy .- mean(lagy, dims=4)

    initest = art(mardata, p)
    ranks = vcat(ranks, p)
    r1, r2, r3, r4, _ = ranks
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

        dlbar1 = dlbarest(cenorig, cenlag, G, U1, U2, U3, U4, U5)
        kronU1 = kron(U5, kron(U4, kron(U3, U2))) * reshape(G, (r1, r2 * r3 * r4 * p))'
        ∇U1 = reshape(dlbar1, (N1, N2 * N1 * N2 * p)) * kronU1
        U1 -= eta * ∇U1
        trackU1[s] = norm(∇U1)

        dlbar2 = dlbarest(cenorig, cenlag, G, U1, U2, U3, U4, U5)
        kronU2 = kron(U5, kron(U4, kron(U3, U1))) * tenmat(G, row=2)'
        ∇U2 = tenmat(dlbar2, row=2) * kronU2
        U2 -= eta * ∇U2
        trackU2[s] = norm(∇U2)

        dlbar3 = dlbarest(cenorig, cenlag, G, U1, U2, U3, U4, U5)
        kronU3 = kron(U5, kron(U4, kron(U2, U1))) * tenmat(G, row=3)'
        ∇U3 = tenmat(dlbar3, row=3) * kronU3
        U3 -= eta * ∇U3
        trackU3[s] = norm(∇U3)

        dlbar4 = dlbarest(cenorig, cenlag, G, U1, U2, U3, U4, U5)
        kronU4 = kron(U5, kron(U3, kron(U2, U1))) * tenmat(G, row=4)'
        ∇U4 = tenmat(dlbar4, row=4) * kronU4
        U4 -= eta * ∇U4
        trackU4[s] = norm(∇U4)

        dlbarG = dlbarest(cenorig, cenlag, G, U1, U2, U3, U4, U5)
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
            U1, U2, U3, U4, U5 = A.fmat
            G = ttm(A.cten, U5', 5)
            U5 = U5'U5
            Arot = full(ttensor(G, [U1, U2, U3, U4, U5]))
            ax = tenmat(Arot, row=[1, 2]) * tenmat(cenlag, row=[1, 2, 3])
            tuckerr = tenmat(cenorig, row=[1, 2]) - ax
            return (G=G, U1=U1, U2=U2, U3=U3, U4=U4, U5=U5, A=Arot, iters=s, fullgrads=fullgrads, residuals=tuckerr)
        end
    end
end

function tuckerreg2(mardata::AbstractArray, ranks::AbstractVector, eta::AbstractFloat=1e-04, maxiter::Int=1000, p::Int=1, ϵ::AbstractFloat=1e-02)
    origy, lagy = tlag(mardata, p)
    cenorig = origy .- mean(origy, dims=3)
    cenlag = lagy .- mean(lagy, dims=4)

    initest = art(mardata, p)
    ranks = vcat(ranks, p)
    hosvdinit = hosvd(initest; reqrank=ranks)
    A = full(hosvdinit)

    U1, U2, U3, U4, U5 = hosvdinit.fmat
    G = ttm(hosvdinit.cten, U5, 5)
    U5 = U5' * U5

    trackG = fill(NaN, maxiter)
    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)

    iters = 0
    for s in ProgressBar(1:maxiter)
        iters += 1

        ∇U1 = ReverseDiff.gradient(x -> objtuckreg(cenorig, cenlag, G, x, U2, U3, U4), U1)
        U1 -= eta * ∇U1
        trackU1[s] = norm(∇U1)

        ∇U2 = ReverseDiff.gradient(x -> objtuckreg(cenorig, cenlag, G, U1, x, U3, U4), U2)
        U2 -= eta * ∇U2
        trackU2[s] = norm(∇U2)

        ∇U3 = ReverseDiff.gradient(x -> objtuckreg(cenorig, cenlag, G, U1, U2, x, U4), U3)
        U3 -= eta * ∇U3
        trackU3[s] = norm(∇U3)

        ∇U4 = ReverseDiff.gradient(x -> objtuckreg(cenorig, cenlag, G, U1, U2, U3, x), U4)
        U4 -= eta * ∇U4
        trackU4[s] = norm(∇U4)

        ∇G = ReverseDiff.gradient(x -> objtuckreg(cenorig, cenlag, x, U1, U2, U3, U4), G)
        trackG[s] = norm(∇G)
        G -= eta * ∇G

        A = full(ttensor(G, [U1, U2, U3, U4, Matrix(U5)]))

        # Stopping Condition
        c = trackU1[s] < ϵ && trackU2[s] < ϵ && trackU3[s] < ϵ && trackU4[s] < ϵ
        if c || s == maxiter
            fullgrads = hcat(trackU1, trackU2, trackU3, trackU4, trackG)
            A = idhosvd(A; reqrank=ranks)
            U1, U2, U3, U4, U5 = A.fmat
            G = ttm(A.cten, U5', 5)
            U5 = U5'U5
            Arot = full(ttensor(G, [U1, U2, U3, U4, U5]))
            ax = tenmat(Arot, row=[1, 2]) * tenmat(cenlag, row=[1, 2, 3])
            tuckerr = tenmat(cenorig, row=[1, 2]) - ax
            return (G=G, U1=U1, U2=U2, U3=U3, U4=U4, U5=U5, A=Arot, iters=s, fullgrads=fullgrads, residuals=tuckerr)
        end
    end
end

