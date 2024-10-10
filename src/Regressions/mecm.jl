
function objmecm(Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)
    obs = size(Y, 2)
    n1 = size(U1, 1)
    n2 = size(U2, 1)
    sigma = -((obs * n1) / 2) * logdet(Σ1) - ((obs * n2) / 2) * logdet(Σ2)
    U2U1 = kron(U2, U1)
    U4U3 = kron(U4, U3)'
    phi12 = kron(ϕ2, ϕ1)
    iΣ21 = inv(kron(Σ2, Σ1))
    ssr = 0
    for i in 3:obs
        phiY = phi12 * (Y[:, (i-1)] - Y[:, (i-2)])
        res = (Y[:, i] - Y[:, i-1]) - U2U1 * U4U3 * Y[:, i-1] - phiY - vec(D)
        ssr += res' * iΣ21 * res
    end
    return sigma - 0.5 * ssr
end

function matobj(Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)
    n1, n2, obs = size(Y)
    U1U3 = U1 * U3'
    U2U4 = U2 * U4'
    iS1 = inv(Σ1)
    iS2 = inv(Σ2)
    sigma = -((obs * n1) / 2) * logdet(Σ1) - ((obs * n2) / 2) * logdet(Σ2)
    ssr = 0
    for i in 3:obs
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, i-2]) * ϕ2'
        ΔY = Y[:, :, i] - Y[:, :, i-1]
        res = ΔY - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        ssr += tr(iS1 * res * iS2 * res')
    end
    return sigma - 0.5 * ssr
end

function mecminit(mardata::AbstractArray, ranks::AbstractVector; p::Int=0)
    ΔY = mardata[:, :, 2:end] - mardata[:, :, 1:(end-1)]
    Y = mardata[:, :, 1:(end-1)]
    N1, N2, _ = size(ΔY)
    initestcoint = tensorols(ΔY, Y)

    if p != 0
        initestmar = tensorols(ΔY, Y)
        permutedmar = permutedims(initestmar, (1, 3, 2, 4))
        vϕ1, _, vϕ2 = svd(tenmat(permutedmar, row=[1, 2]))
        ϕ1 = reshape(vϕ1[:, 1], N1, N1)
        ϕ2 = reshape(vϕ2[:, 1], N2, N2)
    else
        ϕ1 = zeros(N1, N1)
        ϕ2 = zeros(N2, N2)
    end

    permA = tenmat(permutedims(initestcoint, (2, 4, 1, 3)), row=[1, 2])
    ll, _, rr = svd(permA)
    left = reshape(ll[:, 1], N2, N2)
    right = reshape(rr[:, 1], N1, N1)
    U2left, _, U4left = svd(left)
    U1right, _, U3right = svd(right)
    U1 = U1right[:, 1:ranks[1]]
    U2 = U2left[:, 1:ranks[2]]
    U3 = U3right[:, 1:ranks[1]]
    U4 = U4left[:, 1:ranks[2]]
    D = zeros(size(ΔY, 1), size(ΔY, 2))

    return (; U1, U2, U3, U4, D, ϕ1, ϕ2)
end

"""
    mecm(mardata::AbstractArray, ranks::AbstractVector; p::Int=0, maxiter::Int=500, ϵ::AbstractFloat=1e-03)

Performs gradient descent optimization for a Matrix Error Correction Model (MECM), iteratively adjusting the estimated parameters to minimize the objective function. The function operates on matrix autoregressive data and returns the optimized matrices along with convergence details.

# Arguments
- `mardata::AbstractArray`: 3D array representing the matrix autoregressive (MAR) data with dimensions (N1, N2, T), where N1 and N2 are the dimensions of each matrix, and T is the number of observations (time points).
- `ranks::AbstractVector`: A vector of length 2 specifying the ranks for matrices U1, U2, U3, and U4 used in the factorization of the model.
  
# Keyword Arguments
- `p::Int=0`: The lag order of the autoregressive process. If p > 0, the model includes autoregressive terms.
- `maxiter::Int=500`: Maximum number of iterations for the gradient descent algorithm.
- `ϵ::AbstractFloat=1e-03`: Convergence threshold for the gradient descent. The algorithm stops when the difference between consecutive objective values is smaller than this value.

# Returns
- `U1, U2, U3, U4`: Estimated factors from the factorization of the matrices.
- `D`: Estimated matrix of intercept terms.
- `ϕ1, ϕ2`: Autoregressive coefficient matrices (if `p > 0`).
- `iters::Int`: Number of iterations performed.
- `fullgrads::Matrix`: A matrix tracking the learning rates for U1, U2, U3, U4, and D during each iteration.
- `converged::Bool`: Boolean indicating whether the optimization converged.
- `llist::Vector`: A vector containing the objective function values across iterations.

# Description
The function initializes the matrices (U1, U2, U3, U4, D, ϕ1, ϕ2) based on the provided data, then iteratively updates these parameters using gradient descent. At each iteration, it computes gradients with respect to each parameter and updates the parameters accordingly. The learning rate for each update is dynamically adjusted using the Hessians of the gradients.

If autoregressive terms (`p > 0`) are specified, additional gradient updates for `ϕ1` and `ϕ2` are performed.

The function tracks the progress of the optimization by storing the learning rates, objective function values, and the number of iterations. The optimization stops when convergence is reached (based on the threshold `ϵ`) or when the maximum number of iterations is exceeded.
"""
function mecm(
    mardata::AbstractArray,
    ranks::AbstractVector;
    p::Int=0,
    maxiter::Int=500,
    etaS::AbstractFloat=1e-09,
    ϵ::AbstractFloat=1e-03
)
    if length(ranks) != 2
        error("ranks must be a vector of length 2")
    end

    U1, U2, U3, U4, D, ϕ1, ϕ2 = mecminit(mardata, ranks; p)
    N1, N2, obs = size(mardata)

    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)
    trackD = fill(NaN, maxiter)
    trackϕ1 = fill(NaN, maxiter)
    trackϕ2 = fill(NaN, maxiter)
    llist = fill(NaN, maxiter)
    Σ1, Σ2 = diagm(ones(N1)), diagm(ones(N2))
    oldobj = matobj(mardata, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)

    iters = 0

    for k in 0:9
        newΣ1 = (0.1 * k + 0.01) * I(N1)
        newΣ2 = (0.1 * k + 0.01) * I(N2)
        for _ in 1:5

            ∇D = mecmsumres(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
            etaD = 1 / spectralradius((obs) * kron(inv(newΣ2), inv(newΣ1)))
            D += etaD * ∇D

            ∇U1 = U1grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
            hU1 = U1hessian(mardata, U2, U3, U4, newΣ1, newΣ2)
            etaU1 = 1 / spectralradius(hU1)
            U1 += etaU1 * ∇U1

            ∇U3 = U3grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
            hU3 = U3hessian(mardata, U1, U2, U4, newΣ1, newΣ2)
            etaU3 = 1 / spectralradius(hU3)
            U3 += etaU3 * ∇U3
            U3 /= U3[1:ranks[1], 1:ranks[1]]

            ∇newΣ1 = Σ1grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
            newΣ1unscaled = newΣ1 + etaS * ∇newΣ1
            newΣ1 = newΣ1unscaled ./ norm(newΣ1unscaled)
            # preΣ1 = Σ1 + etaS * ∇Σ1
            # eΣ1 = eigen(preΣ1)
            # Σ1 = eΣ1.vectors * diagm(max.(eΣ1.values, 0)) * eΣ1.vectors' + 1e-06I

            ∇U2 = U2grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
            hU2 = U2hessian(mardata, U1, U3, U4, newΣ1, newΣ2)
            etaU2 = 1 / spectralradius(hU2)
            U2 += etaU2 * ∇U2

            ∇U4 = U4grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
            hU4 = U4hessian(mardata, U1, U2, U3, newΣ1, newΣ2)
            etaU4 = 1 / spectralradius(hU4)
            U4 += etaU4 * ∇U4
            U4 /= U4[1:ranks[2], 1:ranks[2]]

            ∇newΣ2 = Σ2grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
            newΣ2 += etaS * ∇newΣ2
            # preΣ2 = Σ2 + etaS * ∇Σ2
            # eΣ2 = eigen(preΣ2)
            # Σ2 = eΣ2.vectors * diagm(max.(eΣ2.values, 0)) * eΣ2.vectors' + 1e-06I

            if p != 0
                ∇ϕ1 = ϕ1grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
                hϕ1 = ϕ1hessian(mardata, ϕ2, newΣ1, newΣ2)
                etaϕ1 = 1 / spectralradius(hϕ1)
                ϕ1unscaled = ϕ1 + etaϕ1 * ∇ϕ1
                ϕ1 = ϕ1unscaled ./ norm(ϕ1unscaled)

                ∇ϕ2 = ϕ2grad(mardata, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2, D)
                hϕ2 = ϕ2hessian(mardata, ϕ1, newΣ1, newΣ2)
                etaϕ2 = 1 / spectralradius(hϕ2)
                ϕ2 += etaϕ2 * ∇ϕ2
            end
        end
        newobj = matobj(mardata, D, U1, U2, U3, U4, newΣ1, newΣ2, ϕ1, ϕ2)
        if newobj > oldobj
            Σ1 .= newΣ1
            Σ2 .= newΣ2
            oldobj = matobj(mardata, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)
        end
    end

    for s in 1:maxiter
        iters += 1

        ∇D = mecmsumres(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
        etaD = 1 / spectralradius((obs) * kron(inv(Σ2), inv(Σ1)))
        D += etaD * ∇D
        trackD[s] = etaD

        ∇U1 = U1grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
        hU1 = U1hessian(mardata, U2, U3, U4, Σ1, Σ2)
        etaU1 = 1 / spectralradius(hU1)
        U1 += etaU1 * ∇U1
        trackU1[s] = etaU1

        ∇U3 = U3grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
        hU3 = U3hessian(mardata, U1, U2, U4, Σ1, Σ2)
        etaU3 = 1 / spectralradius(hU3)
        U3 += etaU3 * ∇U3
        U3 /= U3[1:ranks[1], 1:ranks[1]]
        trackU3[s] = etaU3

        ∇Σ1 = Σ1grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
        Σ1unscaled = Σ1 + etaS * ∇Σ1
        Σ1 = Σ1unscaled ./ norm(Σ1unscaled)
        # preΣ1 = Σ1 + etaS * ∇Σ1
        # eΣ1 = eigen(preΣ1)
        # Σ1 = eΣ1.vectors * diagm(max.(eΣ1.values, 0)) * eΣ1.vectors' + 1e-06I

        ∇U2 = U2grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
        hU2 = U2hessian(mardata, U1, U3, U4, Σ1, Σ2)
        etaU2 = 1 / spectralradius(hU2)
        U2 += etaU2 * ∇U2
        trackU2[s] = etaU2

        ∇U4 = U4grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
        hU4 = U4hessian(mardata, U1, U2, U3, Σ1, Σ2)
        etaU4 = 1 / spectralradius(hU4)
        U4 += etaU4 * ∇U4
        U4 /= U4[1:ranks[2], 1:ranks[2]]
        trackU4[s] = etaU4

        ∇Σ2 = Σ2grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
        Σ2 += etaS * ∇Σ2
        # preΣ2 = Σ2 + etaS * ∇Σ2
        # eΣ2 = eigen(preΣ2)
        # Σ2 = eΣ2.vectors * diagm(max.(eΣ2.values, 0)) * eΣ2.vectors' + 1e-06I

        if p != 0
            ∇ϕ1 = ϕ1grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
            hϕ1 = ϕ1hessian(mardata, ϕ2, Σ1, Σ2)
            etaϕ1 = 1 / spectralradius(hϕ1)
            ϕ1unscaled = ϕ1 + etaϕ1 * ∇ϕ1
            ϕ1 = ϕ1unscaled ./ norm(ϕ1unscaled)
            trackϕ1[s] = norm(∇ϕ1)

            ∇ϕ2 = ϕ2grad(mardata, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
            hϕ2 = ϕ2hessian(mardata, ϕ1, Σ1, Σ2)
            etaϕ2 = 1 / spectralradius(hϕ2)
            ϕ2 += etaϕ2 * ∇ϕ2
            trackϕ2[s] = norm(∇ϕ2)
        end
        llist[s] = matobj(mardata, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)

        if s > 1
            ∇diff = abs(llist[s] - llist[s-1])
            converged = (s == maxiter)

            if (∇diff < ϵ) || converged || (llist[s] < llist[s-1])
                fullgrads = hcat(trackU1, trackU2, trackU3, trackU4, trackD)
                converged = (!converged)
                return (; U1, U2, U3, U4, D, Σ1, Σ2, ϕ1, ϕ2, iters, fullgrads, converged, llist)
            end
        end
    end
end

function mecm2(
    mardata::AbstractArray,
    ranks::AbstractVector;
    p::Int=0,
    eta::Real=1e-02,
    maxiter::Int=500,
    ϵ::AbstractFloat=1e-03,
)
    if length(ranks) != 2
        error("ranks must be a vector of length 2")
    end

    U1, U2, U3, U4, D, ϕ1, ϕ2, Σ1, Σ2 = mecminit(mardata, ranks; p)
    my = tenmat(mardata, row=[1, 2])

    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)
    trackD = fill(NaN, maxiter)
    trackϕ1 = fill(NaN, maxiter)
    trackϕ2 = fill(NaN, maxiter)
    llist = fill(NaN, maxiter)

    iters = 0
    for s in 1:maxiter

        iters += 1

        ∇D = gradient(x -> objmecm(my, x, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), D)[1]
        hD = hessian(x -> objmecm(my, x, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), D)
        etaD = 1 / (maximum(abs.(eigvals(hD))))
        D += etaD * ∇D
        trackD[s] = etaD

        ∇U1 = gradient(x -> objmecm(my, D, x, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U1)[1]
        hU1 = hessian(x -> objmecm(my, D, x, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U1)
        etaU1 = 1 / (maximum(abs.(eigvals(hU1))))
        U1 += etaU1 * ∇U1
        trackU1[s] = etaU1

        ∇U2 = gradient(x -> objmecm(my, D, U1, x, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U2)[1]
        hU2 = hessian(x -> objmecm(my, D, U1, x, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U2)
        etaU2 = 1 / (maximum(abs.(eigvals(hU2))))
        U2 += etaU2 * ∇U2
        trackU2[s] = etaU2

        ∇U3 = gradient(x -> objmecm(my, D, U1, U2, x, U4, Σ1, Σ2, ϕ1, ϕ2), U3)[1]
        hU3 = hessian(x -> objmecm(my, D, U1, U2, x, U4, Σ1, Σ2, ϕ1, ϕ2), U3)
        etaU3 = 1 / (maximum(abs.(eigvals(hU3))))
        U3 += etaU3 * ∇U3
        trackU3[s] = etaU3

        ∇U4 = gradient(x -> objmecm(my, D, U1, U2, U3, x, Σ1, Σ2, ϕ1, ϕ2), U4)[1]
        hU4 = hessian(x -> objmecm(my, D, U1, U2, U3, x, Σ1, Σ2, ϕ1, ϕ2), U4)
        etaU4 = 1 / (maximum(abs.(eigvals(hU4))))
        U4 += etaU4 * ∇U4
        trackU4[s] = etaU4

        if p != 0
            ∇ϕ1 = gradient(x -> objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, x, ϕ2), ϕ1)[1]
            ϕ1 += eta * ∇ϕ1
            trackϕ1[s] = norm(∇ϕ1)

            ∇ϕ2 = gradient(x -> objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, x), ϕ2)[1]
            ϕ2 += eta * ∇ϕ2
            trackϕ2[s] = norm(∇ϕ2)
        end
        llist[s] = objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)

        # Stopping Condition
        if s > 1
            ∇diff = abs(llist[s] - llist[s-1])

            converged = (s == maxiter)

            if (∇diff < ϵ) || converged
                fullgrads = hcat(trackU1, trackU2, trackU3, trackU4, trackD)
                converged = (!converged)
                return (; U1, U2, U3, U4, D, ϕ1, ϕ2, iters, fullgrads, converged, llist)
            end
        end
    end
end


