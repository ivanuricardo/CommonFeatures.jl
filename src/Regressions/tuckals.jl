using TensorEconometrics, TensorToolbox, LinearAlgebra, Random

dimvals = [6, 5]
ranks = [5, 4, 3, 2]
obs = 500
scale = 5
p = 1
maxeigen = 0.9
maxiters = 50
ϵ = 1e-06

marsim = simulatetuckerdata(dimvals, ranks, obs, scale, p, maxeigen)
mardata = marsim.data

function tuckals(mardata, ranks, ϵ, maxiters)
    initest = art(mardata, p)
    origy, lagy = tlag(mardata, p)

    hosvdinit = hosvd(initest; reqrank=ranks)

    Gnew = hosvdinit.cten
    U1new = hosvdinit.fmat[1]
    U2new = hosvdinit.fmat[2]
    U3new = hosvdinit.fmat[3]
    U4new = hosvdinit.fmat[4]

    trackU1 = fill(NaN, maxiters)
    trackU2 = fill(NaN, maxiters)
    trackU3 = fill(NaN, maxiters)
    trackU4 = fill(NaN, maxiters)

    converged = false

    niters = 0
    for s in 1:maxiters
        niters += 1
        U1old = U1new
        omitten1 = ttm(ttm(ttm(Gnew, U2new, 2), U3new, 3), U4new, 4)
        oo1 = contract(omitten1, [3, 4], lagy, [1, 2])
        matoo1 = tenmat(oo1, row=1)
        U1new = tenmat(origy, row=1) * matoo1' * inv(matoo1 * matoo1')
        trackU1[s] = norm(U1old - U1new)
        conv1 = norm(U1old - U1new)
        if conv1 < ϵ
            converged = true
            break
        end

        U2old = U2new
        omitten2 = ttm(ttm(ttm(Gnew, U1new, 1), U3new, 3), U4new, 4)
        oo2 = contract(omitten2, [3, 4], lagy, [1, 2])
        matoo2 = tenmat(oo2, row=2)
        U2new = tenmat(origy, row=2) * matoo2' * inv(matoo2 * matoo2')
        conv2 = norm(U2old - U2new)
        trackU2[s] = norm(U2old - U2new)
        if conv2 < ϵ
            converged = true
            break
        end

        U3old = U3new
        omitten3 = ttm(ttm(ttm(Gnew, U1new, 1), U2new, 2), U4new, 4)
        hh3 = contract(omitten3, 4, lagy, 2)
        mathh3 = tenmat(hh3, row=[3, 4])
        vecU3 = vec(origy)' * mathh3' * inv(mathh3 * mathh3')
        U3new = reshape(vecU3, size(U3new))
        trackU3[s] = norm(U3old - U3new)
        conv3 = norm(U3old - U3new)
        if conv3 < ϵ
            converged = true
            break
        end

        U4old = U4new
        omitten4 = ttm(ttm(ttm(Gnew, U1new, 1), U2new, 2), U3new, 3)
        hh4 = contract(omitten4, 3, lagy, 1)
        mathh4 = tenmat(hh4, row=[3, 4])
        vecU4 = vec(origy)' * mathh4' * inv(mathh4 * mathh4')
        U4new = reshape(vecU4, size(U4new))
        trackU4[s] = norm(U4old - U4new)
        conv4 = norm(U4old - U4new)
        if conv4 < ϵ
            converged = true
            break
        end

        Gold = Gnew
        xstar = ttm(ttm(lagy, U1new', 1), U2new', 2)
        ystar = ttm(ttm(origy, pinv(U3new), 1), pinv(U4new), 2)
        unpermed = tensorols(ystar, xstar)
        Gnew = permutedims(unpermed, [3, 4, 1, 2])
        convG = norm(Gold - Gnew)
        if convG < ϵ
            converged = true
            break
        end
    end
    rebuildA = full(ttensor(Gnew, [U1new, U2new, Matrix(U3new), Matrix(U4new)]))
    hosvdA = hosvd(rebuildA; reqrank=ranks)
    fullgrads = hcat(trackU1, trackU2, trackU3, trackU4)
    return (A=full(hosvdA), G=hosvdA.cten, U1=hosvdA.fmat[1], U2=hosvdA.fmat[2],
        U3=hosvdA.fmat[3], U4=hosvdA.fmat[4], fullgrads=fullgrads)

end

