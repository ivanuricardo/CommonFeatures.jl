
function rorth(n1::Int, n2::Int)
    A = randn(n1, n2)
    Q, _ = qr(A)
    return Matrix(Q)
end

function generatemecmparams(n, ranks; genphi=false)
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

    u2, _, u4 = svd(A)
    u1, _, u3 = svd(B)
    U1 = u1[:, 1:ranks[1]]
    U2 = u2[:, 1:ranks[2]]
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

function generatemecmdata(U1, U2, U3, U4, ϕ1, ϕ2, obs; burnin=100, snr::Real=0.7, matrixnorm=true)
    n1 = size(ϕ1, 1)
    n2 = size(ϕ2, 1)
    Y = zeros(n1 * n2, obs + burnin)
    kron21 = kron(U2, U1)
    kron43 = kron(U4, U3)
    kronphi = kron(ϕ2, ϕ1)
    compmat = mecmstable(U1, U2, U3, U4, ϕ1, ϕ2)
    rho = maximum(compmat)
    diagerr = repeat([rho / snr], n1 * n2)

    if matrixnorm
        Σ1 = diagm(sqrt.(diagerr[1:n1])) ./ norm(diagm(sqrt.(diagerr[1:n1])))
        Σ2 = diagm(sqrt.(diagerr[1:n2]))
        d = MatrixNormal(zeros(n1, n2), Σ1, Σ2)
    else
        Σ = diagm(diagerr)
        d = MultivariateNormal(zeros(n1 * n2), Σ)
    end

    for i in 3:(obs+burnin)
        piy = kron21 * kron43' * Y[:, i-1]
        phiy = kronphi * (Y[:, i-1] - Y[:, i-2])
        Y[:, i] .= Y[:, i-1] + piy + phiy + vec(rand(d))
    end
    data = matten(Y[:, (burnin+1):end], [1, 2], [3], [n1, n2, obs])

    mdy = data[:, :, 2:end] - data[:, :, 1:end-1]
    my = data[:, :, 1:end-1]
    N1, N2, obs = size(my)
    flatdata = Y[:, (burnin+1):end]
    D = zeros(size(mdy, 1) * size(mdy, 2))
    if matrixnorm
        ll = objmecm(flatdata, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)
        return (; data, flatdata, ll, Σ1, Σ2)
    else
        ll = objmecm(flatdata, D, U1, U2, U3, U4, I(N1), I(N2), ϕ1, ϕ2)
        return (; data, flatdata, ll, Σ)
    end

end

function selectmecm(data; p=0, maxiter=50, ϵ=1e-02, etaS=1e-08)
    n1, n2, obs = size(data)
    grid = collect(Iterators.product(1:n1, 1:n2))
    ictable = fill(NaN, 5, n1 * n2)

    for i in ProgressBar(1:(n1*n2))
        selectedrank = collect(grid[i])
        numpars = cointpar([n1, n2], selectedrank)
        mecmest = mecm(data, selectedrank; p, maxiter, etaS, ϵ)
        loglike = mecmest.llist[findlast(!isnan, mecmest.llist)]
        ictable[1, i] = mecmaic(loglike, numpars)
        ictable[2, i] = mecmbic(loglike, numpars, obs)
        ictable[3, i] = mecmhqc(loglike, numpars, obs)
        ictable[4, i] = selectedrank[1]
        ictable[5, i] = selectedrank[2]
    end

    aicvec = argmin(ictable[1, :])
    aicsel = Int.(ictable[4:end, aicvec])
    bicvec = argmin(ictable[2, :])
    bicsel = Int.(ictable[4:end, bicvec])
    hqcvec = argmin(ictable[3, :])
    hqcsel = Int.(ictable[4:end, hqcvec])
    return (; aicsel, bicsel, hqcsel, ictable)
end



