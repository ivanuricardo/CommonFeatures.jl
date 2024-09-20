
function dlbarcoint(origy, lagy, G, U::Vector{<:AbstractMatrix})
    A = full(ttensor(G, [U[1], U[2], U[3], U[4]]))
    N1, N2, obs = size(lagy)
    dlbar = zeros(N1, N2, N1, N2)
    innert = zeros(N1, N2)
    for i in 1:obs
        tena = reshape(A, (N1 * N2, N1 * N2))
        innert = reshape(tena * vec(lagy[:, :, i]), (N1, N2))
        dlbar += ttt((innert - origy[:, :, i]), lagy[:, :, i])
    end
    dlbar .= dlbar ./ (obs)
end

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

function orth!(A::AbstractArray)
    A .= Matrix(qr(A).Q)
end

function mecm(
    mardata::AbstractArray,
    ranks::AbstractVector;
    p::Int=0,
    eta::Real=1e-02,
    maxiter::Int=500,
    ϵ::AbstractFloat=1e-03
)
    if length(ranks) != 2
        error("ranks must be a vector of length 2")
    end

    mdy = mardata[:, :, 2:end] - mardata[:, :, 1:end-1]
    my = mardata[:, :, 1:end-1]
    N1, N2, obs = size(my)
    # initestcoint = tensorols(mdy, my)

    if p != 0
        initestmar = tensorols(mdy[:, :, 2:end], my[:, :, 1:end-1])
        vϕ1, _, vϕ2 = svd(tenmat(permutedims(initestmar, (1, 3, 2, 4)), row=[1, 2]))
        ϕ1 = reshape(vϕ1[:, 1], N1, N1)
        ϕ2 = reshape(vϕ2[:, 1], N2, N2)
        trackϕ1 = fill(NaN, maxiter)
        trackϕ2 = fill(NaN, maxiter)
    else
        ϕ1 = zeros(N1, N1)
        ϕ2 = zeros(N2, N2)
    end

    ΔY = reshape(mdy, N1 * N2, obs)
    Y = reshape(my, N1 * N2, obs)
    D = zeros(size(mdy, 1) * size(mdy, 2))

    # Alternative initialization
    U1 = zeros(N1, ranks[1])
    U2 = zeros(N2, ranks[2])
    U3 = zeros(N1, ranks[1])
    U4 = zeros(N2, ranks[2])
    Σ1, Σ2 = I(N1), I(N2)
    # basell = objmecm(ΔY, Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)

    # for _ in ProgressBar(1:initcount)
    #     newU1 = 0.1 .* randn(N1, ranks[1])
    #     newU2 = 0.1 .* randn(N2, ranks[2])
    #     newU3 = 0.1 .* randn(N1, ranks[1])
    #     newU4 = 0.1 .* randn(N2, ranks[2])
    #     newD = 0.1 .* randn(size(mdy, 1) * size(mdy, 2))
    #     for _ in 1:warmstart
    #         ∇U1 = gradient(x -> objmecm(ΔY, Y, newD, x, newU2, newU3, newU4, Σ1, Σ2, ϕ1, ϕ2), newU1)[1]
    #         newU1 += eta * ∇U1
    #
    #         ∇U2 = gradient(x -> objmecm(ΔY, Y, newD, newU1, x, newU3, newU4, Σ1, Σ2, ϕ1, ϕ2), newU2)[1]
    #         newU2 += eta * ∇U2
    #
    #         ∇U3 = gradient(x -> objmecm(ΔY, Y, newD, newU1, newU2, x, newU4, Σ1, Σ2, ϕ1, ϕ2), newU3)[1]
    #         newU3 += eta * ∇U3
    #
    #         ∇U4 = gradient(x -> objmecm(ΔY, Y, newD, newU1, newU2, newU3, x, Σ1, Σ2, ϕ1, ϕ2), newU4)[1]
    #         newU4 += eta * ∇U4
    #
    #         ∇D = gradient(x -> objmecm(ΔY, Y, x, newU1, newU2, newU3, newU4, Σ1, Σ2, ϕ1, ϕ2), newD)[1]
    #         newD += eta * ∇D
    #
    #         if p != 0
    #             ∇ϕ1 = gradient(x -> objmecm(ΔY, Y, D, U1, U2, U3, U4, Σ1, Σ2, x, ϕ2), ϕ1)[1]
    #             ϕ1 += eta * ∇ϕ1
    #
    #             ∇ϕ2 = gradient(x -> objmecm(ΔY, Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, x), ϕ2)[1]
    #             ϕ2 += eta * ∇ϕ2
    #         end
    #     end
    #     newll = objmecm(ΔY, Y, newD, newU1, newU2, newU3, newU4, Σ1, Σ2, ϕ1, ϕ2)
    #     if newll > basell
    #         U1 .= newU1
    #         U2 .= newU2
    #         U3 .= newU3
    #         U4 .= newU4
    #         basell .= newll
    #     end
    # end

    # permA = tenmat(permutedims(initestcoint, (2, 4, 1, 3)), row=[1, 2])
    # ll, _, rr = svd(permA)
    # left = reshape(ll[:, 1], N2, N2)
    # right = reshape(rr[:, 1], N1, N1)
    # U2left, _, U4left = svd(left)
    # U1right, _, U3right = svd(right)
    # U1 = U1right[:, 1:ranks[1]]
    # U2 = U2left[:, 1:ranks[2]]
    # U3 = U3right[:, 1:ranks[1]]
    # U4 = U4left[:, 1:ranks[2]]

    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)
    trackD = fill(NaN, maxiter)
    llist = fill(NaN, maxiter)

    iters = 0
    for s in ProgressBar(1:maxiter)
        # for s in 1:maxiter

        iters += 1

        ∇U1 = gradient(x -> objmecm(ΔY, Y, D, x, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U1)[1]
        U1 += eta * ∇U1
        trackU1[s] = norm(∇U1)

        ∇U2 = gradient(x -> objmecm(ΔY, Y, D, U1, x, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U2)[1]
        U2 += eta * ∇U2
        trackU2[s] = norm(∇U2)

        ∇U3 = gradient(x -> objmecm(ΔY, Y, D, U1, U2, x, U4, Σ1, Σ2, ϕ1, ϕ2), U3)[1]
        U3 += eta * ∇U3
        trackU3[s] = norm(∇U3)

        ∇U4 = gradient(x -> objmecm(ΔY, Y, D, U1, U2, U3, x, Σ1, Σ2, ϕ1, ϕ2), U4)[1]
        U4 += eta * ∇U4
        trackU4[s] = norm(∇U4)

        ∇D = gradient(x -> objmecm(ΔY, Y, x, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), D)[1]
        D += eta * ∇D
        trackD[s] = norm(∇D)

        if p != 0
            ∇ϕ1 = gradient(x -> objmecm(ΔY, Y, D, U1, U2, U3, U4, Σ1, Σ2, x, ϕ2), ϕ1)[1]
            ϕ1 += eta * ∇ϕ1
            trackϕ1[s] = norm(∇ϕ1)

            ∇ϕ2 = gradient(x -> objmecm(ΔY, Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, x), ϕ2)[1]
            ϕ2 += eta * ∇ϕ2
            trackϕ2[s] = norm(∇ϕ2)
        end
        llist[s] = objmecm(ΔY, Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)

        # Stopping Condition
        if s > 1
            ∇diff = abs(llist[s] - llist[s-1])

            converged = (s == maxiter)

            if (∇diff < ϵ) || converged
                fullgrads = hcat(trackU1, trackU2, trackU3, trackU4, trackD)
                U2U1U4U3 = kron(U2, U1) * kron(U4, U3)'
                phi21 = kron(ϕ2, ϕ1)
                res = ΔY - U2U1U4U3 * Y - phi21 * ΔY .- D
                converged = (!converged)
                return (; U1, U2, U3, U4, D, ϕ1, ϕ2, iters, fullgrads, res, converged, llist)
            end
        end
    end
end

function cointgradU(whichU::Int,
    Y::AbstractArray,
    X::AbstractArray,
    G::AbstractArray,
    U::Vector{<:AbstractMatrix})
    dlbar = dlbarcoint(Y, X, G, U)

    kronU_dict = Dict(
        1 => kron(U[4], kron(U[3], U[2])) * tenmat(G, row=1)',
        2 => kron(U[4], kron(U[3], U[1])) * tenmat(G, row=2)',
        3 => kron(U[4], kron(U[2], U[1])) * tenmat(G, row=3)',
        4 => kron(U[3], kron(U[2], U[1])) * tenmat(G, row=4)'
    )

    kronU = kronU_dict[whichU]
    ∇U = tenmat(dlbar, row=whichU) * kronU
    return ∇U
end
"""
    tuckercoint(mardata, ranks::AbstractVector, eta::Real, a=1, b=1, ϵ=1e-04, maxiter=1000, p=1,ϵ=1e-02)

Perform a matrix autoregression with one lag (MAR(1)) with a reduced tucker rank along the coefficients.
Uses the gradient descent algorithm of Wang, Zhang, and Li 2024.

# Arguments
- `mardata::AbstractArray`: Input tensor data. Should be ``N1 \times N2 \times T``
- `ranks::AbstractVector`: Vector specifying the desired ranks of the Tucker decomposition.
- `eta::AbstractFloat`: Learning rate for gradient descent (default: 1e-03).
- `a::Real`: Regularization parameter (default: 1).
- `b::Real`: Regularization parameter (default: 1).
- `maxiter::Int`: Maximum number of iterations for gradient descent (default: 3000).
- `p::Int`: Number of lags to include
- `ϵ::AbstractFloat`: Convergence threshold for stopping criteria (default: 1e-03).

# Returns
A tuple containing the Tucker decomposition components:
- `G`: Core tensor.
- `U1`, `U2`, `U3`, `U4`: Factor matrices.
- `A`: Full Tucker decomposition tensor.
- `iters`: Number of iterations performed.
- `fullgrads`: A matrix keeping track of gradients. Can be plotted to determine whether gradients behave properly.
"""
function mecm2(
    mardata::AbstractArray,
    ranks::AbstractVector;
    eta::Real=1e-03,
    maxiter::Int=500,
    ϵ::AbstractFloat=1e-03,
    initest::AbstractArray=[]
)
    if initest == []
        ΔY = mardata[:, :, 2:end] - mardata[:, :, 1:end-1]
        Y = mardata[:, :, 1:end-1]
        initest = tensorols(ΔY, Y)
    else
        ΔY = mardata[:, :, 2:end] - mardata[:, :, 1:end-1]
        Y = mardata[:, :, 1:end-1]
    end

    hosvdinit = idhosvd(initest, ranks)
    A = full(hosvdinit)

    U1, U2, U3, U4 = hosvdinit.fmat
    U = [U1, U2, U3, U4]
    G = matten(I(ranks[1] * ranks[2]), [1, 2], [3, 4], ranks)

    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)

    iters = 0
    for s in 1:maxiter
        iters += 1

        ∇U1 = cointgradU(1, ΔY, Y, G, U)
        U[1] -= eta * ∇U1
        trackU1[s] = norm(∇U1)

        ∇U2 = cointgradU(2, ΔY, Y, G, U)
        U[2] -= eta * ∇U2
        trackU2[s] = norm(∇U2)

        ∇U3 = cointgradU(3, ΔY, Y, G, U)
        U[3] -= eta * ∇U3
        trackU3[s] = norm(∇U3)

        ∇U4 = cointgradU(4, ΔY, Y, G, U)
        U[4] -= eta * ∇U4
        trackU4[s] = norm(∇U4)

        A = full(ttensor(G, U))

        # Stopping Condition
        if s > 1
            ∇diff = [abs(trackU1[s] - trackU1[s-1]),
                abs(trackU2[s] - trackU2[s-1]),
                abs(trackU3[s] - trackU3[s-1]),
                abs(trackU4[s] - trackU4[s-1])]

            c = all(∇diff .< ϵ)
            converged = (s == maxiter)

            if c || converged
                fullgrads = hcat(trackU1, trackU2, trackU3, trackU4)
                A = full(ttensor(G, [U[1], U[2], U[3], U[4]]))
                ax = tenmat(A, row=[1, 2]) * tenmat(Y, row=[1, 2])
                residuals = tenmat(ΔY, row=[1, 2]) - ax
                iters = s
                converged = (!converged)

                return LowRankTensorAutoRegression(G, U, A, iters, fullgrads, residuals, converged)
            end
        end
    end
end
