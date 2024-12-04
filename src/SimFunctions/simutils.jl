
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
function simstats(selectedvals::AbstractMatrix, correctval::AbstractVector, sims::Int)
    avgval = mean(selectedvals, dims=2)
    stdval = std(selectedvals, dims=2)
    numvals = size(selectedvals, 1)
    freqcorrect = fill(NaN, numvals)
    freqhigh = fill(NaN, numvals)
    freqlow = fill(NaN, numvals)

    for i in 1:numvals
        cval = correctval[i]
        freqcorrect[i] = count(x -> x == cval, selectedvals[i, :]) / sims
        freqhigh[i] = count(x -> x > cval, selectedvals[i, :]) / sims
        freqlow[i] = count(x -> x < cval, selectedvals[i, :]) / sims
    end

    return (; avgval, stdval, freqcorrect, freqhigh, freqlow)
end

function simstats(selectedvals::AbstractVector, correctval::Int, sims::Int)
    avgval = mean(selectedvals)
    stdval = std(selectedvals)

    freqcorrect = count(x -> x == correctval, selectedvals) / sims
    freqhigh = count(x -> x > correctval, selectedvals) / sims
    freqlow = count(x -> x < correctval, selectedvals) / sims

    return (; avgval, stdval, freqcorrect, freqhigh, freqlow)
end
