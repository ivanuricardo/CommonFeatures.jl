
function mecmsumres(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N1, N2, obs = size(ΔY)
    res = zeros(N1, N2)
    U1U3 = U1 * U3'
    U2U4 = U2 * U4'
    for i in 2:obs
        phiY = ϕ1 * ΔY[:, :, (i-1)] * ϕ2'
        res += ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
    end
    return res
end

function U1grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N1, r1 = size(U1)
    obs = size(Y, 3)
    sumtot = zeros(N1, r1)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 2:obs
        U2U4YU3 = U2U4 * Y[:, :, (i-1)]' * U3
        phiY = ϕ1 * ΔY[:, :, (i-1)] * ϕ2'
        res = ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        sumtot += res * U2U4YU3
    end
    return sumtot
end

function U1hessian(Y, U2, U3, U4)
    _, r1 = size(U3)
    totsum = zeros(r1, r1)
    obs = size(Y, 3)
    U4U2U2U4 = U4 * U2' * U2 * U4'
    for i in 1:obs
        totsum += U3' * Y[:, :, (i-1)] * U4U2U2U4 * Y[:, :, (i-1)]' * U3
    end
    return totsum
end

function U2grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N2, r2 = size(U2)
    obs = size(Y, 3)
    sumtot = zeros(N2, r2)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 2:obs
        U4YU3U1 = U1U3 * Y[:, :, (i-1)] * U4
        phiY = ϕ1 * ΔY[:, :, (i-1)] * ϕ2'
        res = ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        sumtot += res' * U4YU3U1
    end
    return sumtot
end

function U2hessian(Y, U1, U3, U4)
    _, r2 = size(U4)
    totsum = zeros(r2, r2)
    obs = size(Y, 3)
    U3U1U1U3 = U3 * U1' * U1 * U3'
    for i in 2:obs
        totsum += U4' * Y[:, :, (i-1)]' * U3U1U1U3 * Y[:, :, (i-1)] * U4
    end
    return totsum
end

function U3grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N1, r1 = size(U3)
    obs = size(Y, 3)
    totsum = zeros(N1, r1)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 2:obs
        premul = Y[:, :, (i-1)] * U2U4'
        phiY = ϕ1 * ΔY[:, :, (i-1)] * ϕ2'
        res = ΔY[:, :, i]' - U2U4 * Y[:, :, (i-1)]' * U1U3' - phiY' - D'
        totsum += premul * res * U1
    end
    return totsum
end

function U3hessian(Y, U1, U2, U4)
    N1, r1 = size(U1)
    obs = size(Y, 3)
    totsum = zeros(N1 * r1, N1 * r1)
    U4U2U2U4 = U4 * U2' * U2 * U4'
    U1U1 = U1'U1
    for i in 2:obs
        totsum += kron(U1U1, Y[:, :, (i-1)] * U4U2U2U4 * Y[:, :, (i-1)]')
    end
    return totsum
end

function U4grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N2, r2 = size(U4)
    obs = size(Y, 3)
    totsum = zeros(N2, r2)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 2:obs
        premul = Y[:, :, (i-1)]' * U1U3'
        phiY = ϕ1 * ΔY[:, :, (i-1)] * ϕ2'
        res = ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        totsum += premul * res * U2
    end
    return totsum
end

function U4hessian(Y, U1, U2, U3)
    N2, r2 = size(U2)
    obs = size(Y, 3)
    totsum = zeros(N2 * r2, N2 * r2)
    U2U2 = U2'U2
    U3U1U1U3 = U3 * U1' * U1 * U3'
    for i in 2:obs
        totsum += kron(U2U2, Y[:, :, (i-1)]' * U3U1U1U3 * Y[:, :, (i-1)])
    end
    return totsum
end

function ϕ1grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N1, _, obs = size(ΔY)
    totsum = zeros(N1, N1)
    U1U3 = U1 * U3'
    U2U4 = U2 * U4'
    for i in 2:obs
        phiY = ϕ1 * ΔY[:, :, (i-1)] * ϕ2'
        res = ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        totsum += res * ϕ2 * ΔY[:, :, i]'
    end
    return totsum
end

function ϕ1hessian(ΔY, ϕ2)
    N1, _, obs = size(ΔY)
    ϕ2ϕ2 = ϕ2'ϕ2
    totsum = zeros(N1, N1)
    for i in 2:obs
        totsum += ΔY[:, :, (i-1)] * ϕ2ϕ2 * ΔY[:, :, (i-1)]'
    end
    return totsum
end

function ϕ2grad(ΔY, Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    _, N2, obs = size(ΔY)
    totsum = zeros(N2, N2)
    U1U3 = U1 * U3'
    U2U4 = U2 * U4'
    for i in 2:obs
        phiY = ϕ1 * ΔY[:, :, (i-1)] * ϕ2'
        res = ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        totsum += res' * ϕ1 * ΔY[:, :, (i-1)]
    end
    return totsum
end

function ϕ2hessian(ΔY, ϕ1)
    _, N2, obs = size(ΔY)
    ϕ1ϕ1 = ϕ1'ϕ1
    totsum = zeros(N2, N2)
    for i in 2:obs
        totsum += ΔY[:, :, (i-1)]' * ϕ1ϕ1 * ΔY[:, :, (i-1)]
    end
    return totsum
end

