
"""
    simstats(selectedranks, correctrank, sims)

Calculate simulation statistics based on selected ranks, correct ranks, and the number of simulations.

# Arguments
- `selectedranks`::AbstractMatrix: Matrix of selected ranks for each simulation. Simulations should be along the rows.
- `correctrank`::AbstractVector: Vector of correct ranks for each simulation.
- `sims`::Int: CNumber of simulations.

# Returns
A named tuple containing the following statistics:
- `avgrank`: Average rank for each simulation.
- `stdrank`: Standard deviation of ranks for each simulation.
- `freqcorrect`: Frequency of correct ranks for each simulation.
- `freqhigh`: Frequency of ranks higher than the correct rank for each simulation.
- `freqlow`: Frequency of ranks lower than the correct rank for each simulation.
"""
function simstats(selectedranks::AbstractMatrix, correctrank::AbstractVector, sims::Int)
    avgrank = mean(selectedranks, dims=2)
    stdrank = std(selectedranks, dims=2)
    freqcorrect = fill(NaN, 4)
    freqhigh = fill(NaN, 4)
    freqlow = fill(NaN, 4)

    for i in 1:4
        crank = correctrank[i]
        freqcorrect[i] = count(x -> x == crank, selectedranks[i, :]) / sims
        freqhigh[i] = count(x -> x > crank, selectedranks[i, :]) / sims
        freqlow[i] = count(x -> x < crank, selectedranks[i, :]) / sims
    end

    return (avgrank=avgrank, stdrank=stdrank, freqcorrect=freqcorrect, freqhigh=freqhigh, freqlow=freqlow)
end

function generaterrvarcoef(N, r, p, maxeigen, facscale=0.5)
    stabit = 0
    Aold = fill(NaN, N, r)
    Bold = fill(NaN, N * p, r)
    Cold = fill(NaN, N, N * p)
    while true
        stabit += 1
        Aold .= facscale .* randn(N, r)
        Bold .= facscale .* randn(N * p, r)
        Cold .= Aold * Bold'

        if isstable(Cold, maxeigen)
            break
        end
    end
    A, _, B = svd(Cold)
    C = A[:, 1:r] * B[:, 1:r]'
    return (A=A[:, 1:r], B=B[:, 1:r], C=C, stabit=stabit)

end

function generatetuckercoef(dimvals, ranks, p, gscale=3, maxeigen=0.9)
    A = fill(NaN, dimvals[1], dimvals[2], dimvals[1], dimvals[2], p)
    G = fill(NaN, ranks[1], ranks[2], ranks[3], ranks[4], p)
    U1 = fill(NaN, dimvals[1], ranks[1])
    U2 = fill(NaN, dimvals[2], ranks[2])
    U3 = fill(NaN, dimvals[1], ranks[3])
    U4 = fill(NaN, dimvals[2], ranks[4])
    U5 = I(p)

    stabit = 0
    while true
        stabit += 1
        unscaledG = randn(ranks[1], ranks[2], ranks[3], ranks[4], p)
        G .= rescaleten(unscaledG, gscale)
        randU1 = randn(dimvals[1], ranks[1])
        randU2 = randn(dimvals[2], ranks[2])
        randU3 = randn(dimvals[1], ranks[3])
        randU4 = randn(dimvals[2], ranks[4])

        U1 .= svd(randU1).U
        U2 .= svd(randU2).U
        U3 .= svd(randU3).U
        U4 .= svd(randU4).U

        hosvdA = ttensor(G, [U1, U2, U3, U4, Matrix(U5)])
        A .= full(hosvdA)
        varA = tenmat(A, row=[1, 2])
        if isstable(varA, maxeigen)
            break
        end
    end
    return (A=A, G=G, U1=U1, U2=U2, U3=U3, U4=U4, U5=U5, stabit=stabit)
end

"""
    simulatetuckerdata(dimvals, ranks, obs, scale)

Simulate Tucker data with specified dimensions, ranks, observation count, and scaling factor.

# Arguments
- `dimvals::AbstractVector`: Dimensions of the tensor (dimvals[1] for the first mode, dimvals[2] for the second mode).
- `ranks::AbstractVector`: Tucker ranks for the four modes.
- `obs::Int`: Number of observations to simulate.
- `A::Array{Float64, 4}`: Coefficient tensor. If `nothing`, a random coefficient tensor will be generated.
- `p::Int`: Number of lags to include. Default is 1 and the maximum is 5.
- `snr::Real`: Desired signal-to-noise ratio. Default is 0.7.

# Returns
A named tuple containing:
- `data::Array{Float64, 3}`: Simulated Tucker data with dimensions (dimvals[1], dimvals[2], obs).
- `A::Array{Float64, 4}`: Chosen coefficient tensor.
- Σ::Array{Float64, 2}: Covariance matrix of the noise.

# Examples
```julia
result = simulatetuckerdata([5, 4], [2, 3, 2, 3], 100, 1.0)
```
"""
function simulatetuckerdata(
    dimvals::AbstractVector,
    ranks::AbstractVector,
    obs::Int,
    A=nothing,
    p::Int=1,
    snr=0.7)
    if isnothing(A)
        A, _, _, _, _, _, _, _ = generatetuckercoef(dimvals, ranks, p)
    end

    rho = spectralradius(makecompanion(tenmat(A, row=[1, 2])))
    Σ = diagm(repeat([rho / snr], dimvals[1] * dimvals[2]))
    d = MultivariateNormal(zeros(dimvals[1] * dimvals[2]), Σ)

    mardata = zeros(dimvals[1], dimvals[2], obs)
    for i in (p+1):obs
        vecϵ = rand(d)
        ϵ = reshape(vecϵ, (dimvals[1], dimvals[2]))

        mardata[:, :, i] .= contract(A[:, :, :, :, 1], [3, 4], mardata[:, :, i-1], [1, 2]) + ϵ

        for j in 2:p
            mardata[:, :, i] .+= contract(A[:, :, :, :, j], [3, 4], mardata[:, :, i-j], [1, 2])
        end
    end

    return (data=mardata, A=A, Σ=Σ)
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
        ϵ = rand(d)
        rrvardata[:, i] .= C * rrvardata[:, i-1] + ϵ
    end
    return (data=rrvardata, stabit=stabit, C=C, A=A, B=B)
end
