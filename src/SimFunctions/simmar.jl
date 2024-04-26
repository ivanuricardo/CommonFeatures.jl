
function generatevarcoef(N, p, maxeigen=0.9, coefscale=0.5)

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
    obs::Int,
    coefscale::Real=0.5,
    maxeigen::Real=0.9,
    burnin::Int=0)

    A, stabit, rho = generatevarcoef(N, p, maxeigen, coefscale)
    data = zeros(N, obs)
    tenA = matten(A, [1], [2, 3], [N, N, p])

    for i in p:obs
        for j in 1:p
            data[:, i] += tenA[:, :, j] * data[:, i-j+1]
        end
        data[:, i] += randn(N)
    end
    return (A=A, data=data[:, burnin:end], stabit=stabit, rho=rho)
end

function generatemarcoef(dimvals::AbstractVector, p, maxeigen::Real=0.9, coefscale=0.5)
    N1 = dimvals[1]
    N2 = dimvals[2]
    N = N1 * N2

    A, stabit, rho = generatevarcoef(N, p, maxeigen, coefscale)
    tenA = matten(A, [1, 2], [3, 4, 5], [N1, N2, N1, N2, p])
    return (A=tenA, stabit=stabit, rho=rho)
end

function generaterrvarcoef(N, r, p, maxeigen, facscale=0.5)
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

"""
    simulatemardata(dimvals, obs, scale)

Simulate matrix autoregressive (MAR) data with specified dimensions, observation count, and scaling factor.

# Arguments
- `dimvals::AbstractVector`: Dimensions of the MAR tensor (dimvals[1] for the first mode, dimvals[2] for the second mode).
- `obs::Int`: Number of observations to simulate.
- `scale::Real`: Scaling factor for the MAR data.

# Returns
A named tuple containing:
- `mardata::Array{Float64, 3}`: Simulated MAR data with dimensions (dimvals[1], dimvals[2], obs).
- `stabit::Int`: Number of iterations required to generate a stable tensor.
- `A::Array{Float64, 4}`: Autoregressive tensor.

# Examples
```julia
result = simulatemardata([5, 4], 100, 1.0)
```
"""
function simulatemardata(dimvals::AbstractVector, obs::Int, facscale::Real, maxeigen::Real=1)
    A = fill(NaN, dimvals[1], dimvals[2], dimvals[1], dimvals[2])
    stabit = 0
    while true
        stabit += 1
        randA = randn(dimvals[1], dimvals[2], dimvals[1], dimvals[2])
        A .= rescaleten(randA, facscale)
        varA = tenmat(A, row=[1, 2])
        if isstable(varA, maxeigen)
            break
        end
    end

    mardata = zeros(dimvals[1], dimvals[2], obs)
    for i in 2:obs
        ϵ = randn(dimvals[1], dimvals[2])
        mardata[:, :, i] .= contract(A, [3, 4], mardata[:, :, i-1], [1, 2]) + ϵ
    end
    return (mardata=mardata, stabit=stabit, A=A)
end


function simulaterrvardata(
    N::Int,
    r::Int,
    obs::Int,
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
        ϵ = rand(N)
        rrvardata[:, i] .= C * rrvardata[:, i-1] + ϵ
    end
    return (data=rrvardata, stabit=stabit, C=C, A=A, B=B)
end
