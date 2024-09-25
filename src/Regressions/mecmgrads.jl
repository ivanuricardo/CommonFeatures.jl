
function mecmsumres(Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    ΔY = Y[:, :, 2:end] - Y[:, :, 1:end-1]
    N1, N2, obs = size(ΔY)
    res = zeros(N1, N2)
    U1U3 = U1 * U3'
    U2U4 = U2 * U4'
    for i in 3:(obs-1)
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)]) * ϕ2'
        res += ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
    end
    return res
end

function U1grad(Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N1, r1 = size(U1)
    ΔY = Y[:, :, 2:end] - Y[:, :, 1:end-1]
    obs = size(ΔY, 3)
    sumtot = zeros(N1, r1)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 3:(obs-1)
        U2U4YU3 = U2U4 * Y[:, :, (i-1)]' * U3
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)]) * ϕ2'
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
    for i in 3:(obs-2)
        totsum += U3' * Y[:, :, (i-1)] * U4U2U2U4 * Y[:, :, (i-1)]' * U3
    end
    return totsum
end

function U2grad(Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N2, r2 = size(U2)
    ΔY = Y[:, :, 2:end] - Y[:, :, 1:end-1]
    obs = size(ΔY, 3)
    sumtot = zeros(N2, r2)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 3:(obs-1)
        U4YU3U1 = U1U3 * Y[:, :, (i-1)] * U4
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)]) * ϕ2'
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
    for i in 3:(obs-2)
        totsum += U4' * Y[:, :, (i-1)]' * U3U1U1U3 * Y[:, :, (i-1)] * U4
    end
    return totsum
end

function U3grad(Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N1, r1 = size(U3)
    ΔY = Y[:, :, 2:end] - Y[:, :, 1:end-1]
    obs = size(ΔY, 3)
    totsum = zeros(N1, r1)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 3:(obs-1)
        premul = Y[:, :, (i-1)] * U2U4'
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)]) * ϕ2'
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
    for i in 3:(obs-2)
        totsum += kron(U1U1, Y[:, :, (i-1)] * U4U2U2U4 * Y[:, :, (i-1)]')
    end
    return totsum
end

function U4grad(Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    N2, r2 = size(U4)
    ΔY = Y[:, :, 2:end] - Y[:, :, 1:end-1]
    obs = size(ΔY, 3)
    totsum = zeros(N2, r2)
    U2U4 = U2 * U4'
    U1U3 = U1 * U3'
    for i in 3:(obs-1)
        premul = Y[:, :, (i-1)]' * U1U3'
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)]) * ϕ2'
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
    for i in 3:(obs-2)
        totsum += kron(U2U2, Y[:, :, (i-1)]' * U3U1U1U3 * Y[:, :, (i-1)])
    end
    return totsum
end

function ϕ1grad(Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    ΔY = Y[:, :, 2:end] - Y[:, :, 1:end-1]
    N1, _, obs = size(ΔY)
    totsum = zeros(N1, N1)
    U1U3 = U1 * U3'
    U2U4 = U2 * U4'
    for i in 3:(obs-1)
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)]) * ϕ2'
        res = ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        totsum += res * ϕ2 * (Y[:, :, (i-1)] - Y[:, :, (i-2)])'
    end
    return totsum
end

function ϕ1hessian(Y, ϕ2)
    N1, _, obs = size(Y)
    ϕ2ϕ2 = ϕ2'ϕ2
    totsum = zeros(N1, N1)
    for i in 3:(obs-2)
        ΔY = Y[:, :, (i-1)] - Y[:, :, (i-2)]
        totsum += ΔY * ϕ2ϕ2 * ΔY'
    end
    return totsum
end

function ϕ2grad(Y, U1, U2, U3, U4, ϕ1, ϕ2, D)
    ΔY = Y[:, :, 2:end] - Y[:, :, 1:end-1]
    _, N2, obs = size(ΔY)
    totsum = zeros(N2, N2)
    U1U3 = U1 * U3'
    U2U4 = U2 * U4'
    for i in 3:(obs-1)
        phiY = ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)]) * ϕ2'
        res = ΔY[:, :, i] - U1U3 * Y[:, :, (i-1)] * U2U4' - phiY - D
        totsum += res' * ϕ1 * (Y[:, :, (i-1)] - Y[:, :, (i-2)])
    end
    return totsum
end

function ϕ2hessian(Y, ϕ1)
    _, N2, obs = size(Y)
    ϕ1ϕ1 = ϕ1'ϕ1
    totsum = zeros(N2, N2)
    for i in 3:(obs-2)
        ΔY = Y[:, :, (i-1)] - Y[:, :, (i-2)]
        totsum += ΔY' * ϕ1ϕ1 * ΔY
    end
    return totsum
end

