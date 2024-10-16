
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
    numranks = size(selectedranks, 1)
    freqcorrect = fill(NaN, numranks)
    freqhigh = fill(NaN, numranks)
    freqlow = fill(NaN, numranks)

    for i in 1:numranks
        crank = correctrank[i]
        freqcorrect[i] = count(x -> x == crank, selectedranks[i, :]) / sims
        freqhigh[i] = count(x -> x > crank, selectedranks[i, :]) / sims
        freqlow[i] = count(x -> x < crank, selectedranks[i, :]) / sims
    end

    return (; avgrank, stdrank, freqcorrect, freqhigh, freqlow)
end
