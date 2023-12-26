function conditionvalue(ten::AbstractArray, ranks)
    unfold1 = tenmat(ten, row=1)
    unfold2 = tenmat(ten, row=2)
    unfold3 = tenmat(ten, row=3)
    unfold4 = tenmat(ten, row=4)
    upperλ = max(opnorm(unfold1), opnorm(unfold2), opnorm(unfold3), opnorm(unfold4))
    lowerλ = min(svd(unfold1).S[ranks[1]], svd(unfold2).S[ranks[2]], svd(unfold3).S[ranks[3]], svd(unfold4).S[ranks[4]])
    conditionnum = upperλ / lowerλ
    return conditionnum
end

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

"""
    simulatetuckerdata(dimvals, ranks, obs, scale)

Simulate Tucker data with specified dimensions, ranks, observation count, and scaling factor.

# Arguments
- `dimvals::AbstractVector`: Dimensions of the tensor (dimvals[1] for the first mode, dimvals[2] for the second mode).
- `ranks::AbstractVector`: Tucker ranks for the four modes.
- `obs::Int`: Number of observations to simulate.
- `scale::Real`: Scaling factor for the Tucker decomposition.
- `p::Int`: Number of lags to include. Default is 1 and the maximum is 5.

# Returns
A named tuple containing:
- `tuckerdata::Array{Float64, 3}`: Simulated Tucker data with dimensions (dimvals[1], dimvals[2], obs).
- `stabit::Int`: Number of iterations required to generate a stable tensor.
- `A::Array{Float64, 4}`: Tucker core tensor.

# Examples
```julia
result = simulatetuckerdata([5, 4], [2, 3, 2, 3], 100, 1.0)
```
"""
function simulatetuckerdata(dimvals::AbstractVector, ranks::AbstractVector, obs::Int, scale::Real=5, p::Int=1, maxeigen=1)
    A = fill(NaN, dimvals[1], dimvals[2], dimvals[1], dimvals[2] * p)
    stabit = 0
    while true
        stabit += 1
        unscaledG = randn(ranks[1], ranks[2], ranks[3], ranks[4])
        G = rescalemat(unscaledG, scale)
        randU1 = randn(dimvals[1], ranks[1])
        randU2 = randn(dimvals[2], ranks[2])
        randU3 = randn(dimvals[1], ranks[3])
        randU4 = randn(dimvals[2] * p, ranks[4])

        U1, _ = svd(randU1)
        U2, _ = svd(randU2)
        U3, _ = svd(randU3)
        U4, _ = svd(randU4)

        hosvdA = ttensor(G, [U1, U2, U3, U4])
        A .= full(hosvdA)
        varA = tenmat(A, row=[1, 2])
        if isstable(varA, maxeigen)
            break
        end
    end

    mardata = zeros(dimvals[1], dimvals[2], obs)
    for i in (p+1):obs
        ϵ = randn(dimvals[1], dimvals[2])
        if p == 1
            mardata[:, :, i] .= contract(A, [3, 4], mardata[:, :, i-1], [1, 2]) + ϵ
        elseif p == 2
            mardata[:, :, i] .= contract(A[:, :, :, 1:dimvals[2]], [3, 4], mardata[:, :, i-1], [1, 2]) + contract(A[:, :, :, (dimvals[2]+1):end], [3, 4], mardata[:, :, i-2], [1, 2]) + ϵ
        elseif p == 3
            mardata[:, :, i] .= contract(A[:, :, :, 1:dimvals[2]], [3, 4], mardata[:, :, i-1], [1, 2]) + contract(A[:, :, :, (dimvals[2]+1):dimvals[2]*2], [3, 4], mardata[:, :, i-2], [1, 2]) + contract(A[:, :, :, (dimvals[2]*2+1):end], [3, 4], mardata[:, :, i-3], [1, 2]) + ϵ
        elseif p == 4
            mardata[:, :, i] .= contract(A[:, :, :, 1:dimvals[2]], [3, 4], mardata[:, :, i-1], [1, 2]) + contract(A[:, :, :, (dimvals[2]+1):dimvals[2]*2], [3, 4], mardata[:, :, i-2], [1, 2]) + contract(A[:, :, :, (dimvals[2]*2+1):(dimvals[2]*3)], [3, 4], mardata[:, :, i-3], [1, 2]) + contract(A[:, :, :, (dimvals[2]*3+1):end], [3, 4], mardata[:, :, i-4], [1, 2]) + ϵ
        elseif p == 5
            mardata[:, :, i] .= contract(A[:, :, :, 1:dimvals[2]], [3, 4], mardata[:, :, i-1], [1, 2]) + contract(A[:, :, :, (dimvals[2]+1):dimvals[2]*2], [3, 4], mardata[:, :, i-2], [1, 2]) + contract(A[:, :, :, (dimvals[2]*2+1):(dimvals[2]*3)], [3, 4], mardata[:, :, i-3], [1, 2]) + contract(A[:, :, :, (dimvals[2]*3+1):(dimvals[2]*4)], [3, 4], mardata[:, :, i-4], [1, 2]) + contract(A[:, :, :, (dimvals[2]*4+1):end], [3, 4], mardata[:, :, i-5], [1, 2]) + ϵ
        end
    end
    return (tuckerdata=mardata, stabit=stabit, A=A, κ=conditionvalue(A, ranks))
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
function simulatemardata(dimvals::AbstractVector, obs::Int, scale::Real)
    A = fill(NaN, dimvals[1], dimvals[2], dimvals[1], dimvals[2])
    stabit = 0
    while true
        stabit += 1
        A .= round.(scale .* randn(dimvals[1], dimvals[2], dimvals[1], dimvals[2]), digits=3)
        varA = tenmat(A, row=[1, 2])
        if isstable(varA)
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

