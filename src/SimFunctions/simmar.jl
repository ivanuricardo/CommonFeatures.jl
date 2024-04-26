
function generatevarcoef(
    N::Int,
    p::Int;
    maxeigen::Real=0.9,
    coefscale::Real=0.5)

    stabit = 0
    A = fill(NaN, N, N * p)

    while true
        stabit += 1
        randA = coefscale .* randn(N, N * p)
        A .= randA
        if isstable(A, maxeigen)
            break
        end
    end
    return (A=A, stabit=stabit, rho=spectralradius(makecompanion(A)))

end

function simulatevardata(N::Int,
    p::Int,
    obs::Int;
    snr::Real=0.7,
    coefscale::Real=0.5,
    maxeigen::Real=0.9,
    A=nothing,
    burnin::Int=1)

    if isnothing(A)
        A, stabit, rho = generatevarcoef(N, p, maxeigen, coefscale)
    else
        _, stabit, rho = generatevarcoef(N, p, maxeigen, coefscale)
    end

    data = zeros(N, obs)
    tenA = matten(A, [1], [2, 3], [N, N, p])

    rho = spectralradius(makecompanion(A))
    Σ = diagm(repeat([rho / snr], N))
    d = MultivariateNormal(zeros(N), Σ)

    for i in (p+1):obs
        for j in 1:p
            data[:, i] += tenA[:, :, j] * data[:, i-j]
        end
        data[:, i] += rand(d)
    end
    return (A=A, data=data[:, burnin:end], stabit=stabit, rho=rho)
end

function generaterrvarcoef(
    N::Int,
    r::Int,
    p::Int;
    maxeigen::Real=0.9,
    facscale::Real=0.5)

    stabit = 0
    A = fill(NaN, N, r)
    B = fill(NaN, N * p, r)
    C = fill(NaN, N, N * p)
    while true
        stabit += 1
        Aold = facscale .* randn(N, r)
        Bold = facscale .* randn(N * p, r)
        Cold = Aold * Bold'

        Ad, _, Bd = svd(Cold)
        A = facscale .* Ad[:, 1:r]
        B = facscale .* Bd[:, 1:r]

        C .= A * B'

        if isstable(C, maxeigen)
            break
        end
    end
    return (A=A[:, 1:r], B=B[:, 1:r], C=C, stabit=stabit)

end

function simulaterrvardata(
    N::Int,
    r::Int,
    obs::Int;
    C=nothing,
    p::Int=1,
    snr::Real=0.7,
    facscale::Real=0.6)
    if isnothing(C)
        A, B, C, stabit = generaterrvarcoef(N, r, p, 0.9, facscale)
    end

    rho = spectralradius(makecompanion(C))
    Σ = diagm(repeat([rho / snr], N))
    d = MultivariateNormal(zeros(N), Σ)

    rrvardata = zeros(N, obs)
    for i in (p+1):obs
        ϵ = rand(d)
        rrvardata[:, i] .= C * rrvardata[:, i-1] + ϵ
    end
    return (data=rrvardata, stabit=stabit, C=C, A=A, B=B)
end
