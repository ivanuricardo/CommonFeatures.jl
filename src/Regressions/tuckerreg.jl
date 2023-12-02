
"""
    tuckerreg(mardata, A, ranks::AbstractVector, eta=1e-04, a=1, b=1, ϵ=1e-04, maxiter=600)

Perform a matrix autoregression with one lag (MAR(1)) with a reduced tucker rank along the coefficients.
Uses the gradient descent algorithm of Wang, Zhang, and Li 2024.

# Arguments
- `mardata::AbstractArray`: Input tensor data. Should be ``N1 \times N2 \times T``
- `ranks::AbstractVector`: Vector specifying the desired ranks of the Tucker decomposition.
- `initest::AbstractArray`: Initial coefficient value (usually MAR)
- `eta::AbstractFloat`: Learning rate for gradient descent (default: 1e-04).
- `a::Real`: Regularization parameter (default: 1).
- `b::Real`: Regularization parameter (default: 1).
- `ϵ::AbstractFloat`: Convergence threshold for stopping criteria (default: 1e-04).
- `maxiter::Int`: Maximum number of iterations for gradient descent (default: 3000).

# Returns
A tuple containing the Tucker decomposition components:
- `G`: Core tensor.
- `U1`, `U2`, `U3`, `U4`: Factor matrices.
- `initest`: Full Tucker decomposition tensor.
- `iters`: Number of iterations performed.
"""
function tuckerreg(mardata::AbstractArray, ranks::AbstractVector, initest::AbstractArray=art(mardata, 1), eta::AbstractFloat=1e-05, a::Real=1, b::Real=1, ϵ::AbstractFloat=1e-01, maxiter::Int=3000)
    origy, lagy = tlag(mardata, 1)
    N1, N2, obs = size(mardata)

    hosvdinit = hosvd(initest; reqrank=ranks)
    Anew = full(hosvdinit)

    Gnew = hosvdinit.cten
    U1new = hosvdinit.fmat[1]
    U2new = hosvdinit.fmat[2]
    U3new = hosvdinit.fmat[3]
    U4new = hosvdinit.fmat[4]

    iters = 0
    for s in 1:maxiter
        iters += 1
        dlbar = zeros(N1, N2, N1, N2)
        innert = zeros(N1, N2)
        for i in 1:(obs-1)
            innert = contract(Anew, [3, 4], lagy[:, :, i], [1, 2])
            dlbar += ttt((innert - origy[:, :, i]), lagy[:, :, i])
        end
        dlbar .= dlbar ./ obs

        clipgradient!(dlbar, 1)

        kronU1 = kron(U4new, kron(U3new, U2new)) * tenmat(Gnew, row=1)'
        regularizeU1 = a * (U1new * (U1new'U1new - (b^2 * I)))
        ∇U1 = tenmat(dlbar, row=1) * kronU1
        U1new -= eta * ∇U1 - eta * regularizeU1

        kronU2 = kron(U4new, kron(U3new, U1new)) * tenmat(Gnew, row=2)'
        regularizeU2 = a * (U2new * (U2new'U2new - (b^2 * I)))
        ∇U2 = tenmat(dlbar, row=2) * kronU2
        U2new -= eta * ∇U2 - eta * regularizeU2

        kronU3 = kron(U4new, kron(U2new, U1new)) * tenmat(Gnew, row=3)'
        regularizeU3 = a * (U3new * (U3new'U3new - (b^2 * I)))
        ∇U3 = tenmat(dlbar, row=3) * kronU3
        U3new -= eta * ∇U3 - eta * regularizeU3

        kronU4 = kron(U3new, kron(U2new, U1new)) * tenmat(Gnew, row=4)'
        regularizeU4 = a * (U4new * (U4new'U4new - (b^2 * I)))
        ∇U4 = tenmat(dlbar, row=4) * kronU4
        U4new -= eta * ∇U4 - eta * regularizeU4

        facmat = [Matrix(U1new'), Matrix(U2new'), Matrix(U3new'), Matrix(U4new')]
        Gnew -= eta * full(ttensor(dlbar, facmat))

        Anew = full(ttensor(Gnew, [U1new, U2new, U3new, U4new]))

        # Stopping Condition
        if norm(∇U1) < ϵ || norm(∇U2) < ϵ || norm(∇U3) < ϵ || norm(∇U4) < ϵ
            A = hosvd(Anew; reqrank=ranks)
            return (G=A.cten, U1=A.fmat[1], U2=A.fmat[2], U3=A.fmat[3],
                U4=A.fmat[4], A=full(A), iters=iters)
        elseif iters == maxiter
            A = hosvd(Anew; reqrank=ranks)
            return (G=A.cten, U1=A.fmat[1], U2=A.fmat[2], U3=A.fmat[3],
                U4=A.fmat[4], A=full(A), iters=iters)
        end
    end
end
