
function rorth(n1::Int, n2::Int)
    A = randn(n1, n2)
    Q, R = qr(A)
    randorth = Q * Diagonal(sign.(diag(R)))
    return randorth
end

function generatemecmparams(n, ranks; genphi=true)
    Q1 = rorth(n[2], ranks[2])
    Q2 = rorth(n[2], ranks[2])
    Q3 = rorth(n[1], ranks[1])
    Q4 = rorth(n[1], ranks[1])
    if genphi
        Qϕ1 = rorth(n[2], n[2])
        Qϕ2 = rorth(n[2], n[2])
        Qϕ3 = rorth(n[1], n[1])
        Qϕ4 = rorth(n[1], n[1])
    end

    A = Q1 * diagm(randn(ranks[2])) * Q2'
    B = Q3 * diagm(randn(ranks[1])) * Q4'

    u2, s1, u4 = svd(A)
    u1, s2, u3 = svd(B)
    U1 = u1[:, 1:ranks[1]] * diagm(s2[1:ranks[1]])
    U2 = u2[:, 1:ranks[2]] * diagm(s1[1:ranks[2]])
    U3 = u3[:, 1:ranks[1]]
    U4 = u4[:, 1:ranks[2]]

    if genphi
        ϕ2 = Qϕ1 * diagm(randn(n[2])) * Qϕ2'
        ϕ1 = Qϕ3 * diagm(randn(n[1])) * Qϕ4'
        return (; U1, U2, U3, U4, ϕ1, ϕ2)
    end
    ϕ1 = zeros(n[1], n[1])
    ϕ2 = zeros(n[2], n[2])

    return (; U1, U2, U3, U4, ϕ1, ϕ2)
end

function mecmstable(U1, U2, U3, U4, ϕ1, ϕ2)
    kronu21 = kron(U2, U1)
    kronu43 = kron(U4, U3)
    kronphi = kron(ϕ2, ϕ1)
    topleft = I + kronu43' * kronu21
    topright = kronu43' * kronphi

    companionmatrix = [topleft topright; kronu21 kronphi]
    return abs.(eigvals(companionmatrix))
end

function generatemecmdata(U1, U2, U3, U4, ϕ1, ϕ2, obs; burnin=100)
    n1 = size(ϕ1, 1)
    n2 = size(ϕ2, 1)
    Y = zeros(n1 * n2, obs + burnin)
    kron21 = kron(U2, U1)
    kron43 = kron(U4, U3)
    kronphi = kron(ϕ2, ϕ1)
    for i in 3:(obs+burnin)
        piy = kron21 * kron43' * Y[:, i-1]
        phiy = kronphi * (Y[:, i-1] - Y[:, i-2])
        Y[:, i] .= Y[:, i-1] + piy + phiy + (0.01 .* randn(n1 * n2))
    end
    data = matten(Y[:, (burnin+1):end], [1, 2], [3], [n1, n2, obs])

    mdy = data[:, :, 2:end] - data[:, :, 1:end-1]
    my = data[:, :, 1:end-1]
    N1, N2, obs = size(my)
    ΔY = Y[:, (burnin+1):end] - Y[:, burnin:(end-1)]
    flatdata = Y[:, burnin:(end-1)]
    D = zeros(size(mdy, 1) * size(mdy, 2))
    ll = objmecm(ΔY, flatdata, D, U1, U2, U3, U4, I(N1), I(N2), ϕ1, ϕ2)
    return (; data, flatdata, ll)
end

function selectmecm(data; p=0, maxiters=50, ϵ=1e-02)
    n1, n2, obs = size(data)
    grid = collect(Iterators.product(1:n1, 1:n2))
    ictable = fill(NaN, 5, n1 * n2)

    for i in 1:(n1*n2)
        selectedrank = collect(grid[i])
        numpars = cointpar([n1, n2], selectedrank)
        mecmest = mecm(data, selectedrank; p=p, maxiter=maxiters, ϵ=ϵ)
        loglike = -mecmest.llist[findlast(!isnan, mecmest.llist)]
        ictable[1, i] = aic(loglike, numpars, obs)
        ictable[2, i] = bic(loglike, numpars, obs)
        ictable[3, i] = hqc(loglike, numpars, obs)
        ictable[4, i] = selectedrank[1]
        ictable[5, i] = selectedrank[2]
    end

    nancols = findall(x -> any(isnan, x), eachcol(ictable))
    filteredic = ictable[:, setdiff(1:size(ictable, 2), nancols)]
    aicvec = argmin(filteredic[1, :])
    aicsel = Int.(filteredic[4:end, aicvec])
    bicvec = argmin(filteredic[2, :])
    bicsel = Int.(filteredic[4:end, bicvec])
    hqcvec = argmin(filteredic[3, :])
    hqcsel = Int.(filteredic[4:end, hqcvec])
    return (; aicsel, bicsel, hqcsel)
end



