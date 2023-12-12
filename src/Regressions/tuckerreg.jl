
"""
    clipgradient!(grad, maxnorm)

Clip the gradient vector in-place to have a maximum Euclidean norm of `maxnorm`.

# Arguments
- `grad::AbstractArray`: The gradient vector to be clipped.
- `maxnorm::Real`: The maximum Euclidean norm allowed for the gradient.

# Returns
- `grad::AbstractArray`: The clipped gradient vector.

# Example
```julia
grad = [1.0, 2.0, 3.0]
maxnorm = 2.5
# In this example, if the Euclidean norm of grad is greater than maxnorm, the vector is scaled to have a norm of maxnorm.
clipgradient!(grad, maxnorm)
```
"""
function clipgradient!(grad, maxnorm)
    normg = norm(grad)
    if normg > maxnorm
        grad .= grad ./ normg * maxnorm
    end

    return grad
end

"""
    tuckerreg(mardata, A, ranks::AbstractVector, eta=1e-04, a=1, b=1, ϵ=1e-04, maxiter=600)

Perform a matrix autoregression with one lag (MAR(1)) with a reduced tucker rank along the coefficients.
Uses the gradient descent algorithm of Wang, Zhang, and Li 2024.

# Arguments
- `mardata::AbstractArray`: Input tensor data. Should be ``N1 \times N2 \times T``
- `ranks::AbstractVector`: Vector specifying the desired ranks of the Tucker decomposition.
- `eta::AbstractFloat`: Learning rate for gradient descent (default: 1e-04).
- `a::Real`: Regularization parameter (default: 1).
- `b::Real`: Regularization parameter (default: 1).
- `ϵ::AbstractFloat`: Convergence threshold for stopping criteria (default: 1e-04).
- `maxiter::Int`: Maximum number of iterations for gradient descent (default: 3000).
- `maxnorm::Real`: Max norm for gradient clipping if required. (default: 1)
- `mineta::AbstractFloat`: If adaptive step size, sets an eta stopping condition if the adaptive step size is too small (default: 1e-20)
- `fixedeta::Bool`: Sets either a fixed or adaptive step size (default: true)
- `orthonorm::Bool`: Bool to normalize columns of factor matrices. (default: true)
- `P::Int`: Number of lags to include

# Returns
A tuple containing the Tucker decomposition components:
- `G`: Core tensor.
- `U1`, `U2`, `U3`, `U4`: Factor matrices.
- `A`: Full Tucker decomposition tensor.
- `iters`: Number of iterations performed.
- `fullgrads`: A matrix keeping track of gradients. Can be plotted to determine whether gradients behave properly.
"""
function tuckerreg(mardata::AbstractArray, ranks::AbstractVector, eta::AbstractFloat=1e-05, a::Real=1, b::Real=1, maxiter::Int=3000, maxnorm::Real=1, fixedeta::Bool=true, orthonorm::Bool=true, p::Int=1)
    initest = art(mardata, p=p)
    origy, lagy = tlag(mardata, p)
    N1, N2, obs = size(mardata)

    hosvdinit = hosvd(initest; reqrank=ranks)
    Anew = full(hosvdinit)

    Gnew = hosvdinit.cten
    U1new = hosvdinit.fmat[1]
    U2new = hosvdinit.fmat[2]
    U3new = hosvdinit.fmat[3]
    U4new = hosvdinit.fmat[4]

    trackU1 = fill(NaN, maxiter)
    trackU2 = fill(NaN, maxiter)
    trackU3 = fill(NaN, maxiter)
    trackU4 = fill(NaN, maxiter)

    iters = 0
    for s in 1:maxiter
        iters += 1
        dlbar = zeros(N1, N2, N1, N2)
        innert = zeros(N1, N2)
        for i in 1:(obs-p)
            innert = contract(Anew, [3, 4], lagy[:, :, i], [1, 2])
            dlbar += ttt((innert - origy[:, :, i]), lagy[:, :, i])
        end
        dlbar .= dlbar ./ obs
        clipgradient!(dlbar, maxnorm)

        kronU1 = kron(U4new, kron(U3new, U2new)) * tenmat(Gnew, row=1)'
        regularizeU1 = a * (U1new * (U1new'U1new - (b^2 * I)))
        ∇U1 = tenmat(dlbar, row=1) * kronU1
        U1new -= eta * ∇U1 - eta * regularizeU1
        trackU1[s] = norm(∇U1)
        if orthonorm == true
            clipgradient!(U1new, maxnorm)
        end

        kronU2 = kron(U4new, kron(U3new, U1new)) * tenmat(Gnew, row=2)'
        regularizeU2 = a * (U2new * (U2new'U2new - (b^2 * I)))
        ∇U2 = tenmat(dlbar, row=2) * kronU2
        U2new -= eta * ∇U2 - eta * regularizeU2
        trackU2[s] = norm(∇U2)
        if orthonorm == true
            clipgradient!(U2new, maxnorm)
        end

        kronU3 = kron(U4new, kron(U2new, U1new)) * tenmat(Gnew, row=3)'
        regularizeU3 = a * (U3new * (U3new'U3new - (b^2 * I)))
        ∇U3 = tenmat(dlbar, row=3) * kronU3
        U3new -= eta * ∇U3 - eta * regularizeU3
        trackU3[s] = norm(∇U3)
        if orthonorm == true
            clipgradient!(U3new, maxnorm)
        end

        kronU4 = kron(U3new, kron(U2new, U1new)) * tenmat(Gnew, row=4)'
        regularizeU4 = a * (U4new * (U4new'U4new - (b^2 * I)))
        ∇U4 = tenmat(dlbar, row=4) * kronU4
        U4new -= eta * ∇U4 - eta * regularizeU4
        trackU4[s] = norm(∇U4)
        if orthonorm == true
            clipgradient!(U4new, maxnorm)
        end

        facmat = [Matrix(U1new'), Matrix(U2new'), Matrix(U3new'), Matrix(U4new')]
        Gnew -= eta * full(ttensor(dlbar, facmat))

        Anew = full(ttensor(Gnew, [U1new, U2new, U3new, U4new]))

        if fixedeta == false
            eta = eta / sqrt(sum(∇U1 .^ 2) + sum(∇U2 .^ 2) + sum(∇U3 .^ 2) + sum(∇U4 .^ 2) + sum(dlbar .^ 2))
        end

        # Stopping Condition
        if s == maxiter
            fullgrads = hcat(trackU1, trackU2, trackU3, trackU4)
            A = hosvd(Anew; reqrank=ranks)
            return (G=A.cten, U1=A.fmat[1], U2=A.fmat[2], U3=A.fmat[3],
                U4=A.fmat[4], A=full(A), iters=iters, fullgrads=fullgrads)
        end
    end
    fullgrads = hcat(trackU1, trackU2, trackU3, trackU4)
    A = hosvd(Anew; reqrank=ranks)

    return (G=A.cten, U1=A.fmat[1], U2=A.fmat[2], U3=A.fmat[3],
        U4=A.fmat[4], A=full(A), iters=iters, fullgrads=fullgrads)
end

function naivetuckreg(mardata, ranks, P)
    initest = art(mardata, P)
    hosvdinit = hosvd(initest; reqrank=ranks)
    return (G=hosvdinit.cten, U1=hosvdinit.fmat[1], U2=hosvdinit.fmat[2], U3=hosvdinit.fmat[3], U4=hosvdinit.fmat[4], A=full(hosvdinit))
end
