
function generatetuckercoef(dimvals, ranks, p; gscale=3, maxeigen=0.9)
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
        U1 .= svd(randU1).U
        randU2 = randn(dimvals[2], ranks[2])
        U2 .= svd(randU2).U
        randU3 = randn(dimvals[1], ranks[3])
        randU4 = randn(dimvals[2], ranks[4])

        U3 .= svd(randU3).U
        U4 .= svd(randU4).U

        hosvdA = ttensor(G, [U1, U2, U3, U4, Matrix(U5)])
        A .= full(hosvdA)
        varA = tenmat(A, row=[1, 2])
        if isstable(varA; maxeigen)
            break
        end
    end
    return (; A, G, U1, U2, U3, U4, U5, stabit)
end
function generatetuckercoef(dimvals, ranks, p, U1; gscale=3, maxeigen=0.9)
    A = fill(NaN, dimvals[1], dimvals[2], dimvals[1], dimvals[2], p)
    G = fill(NaN, ranks[1], ranks[2], ranks[3], ranks[4], p)
    U2 = fill(NaN, dimvals[2], ranks[2])
    U3 = fill(NaN, dimvals[1], ranks[3])
    U4 = fill(NaN, dimvals[2], ranks[4])
    U5 = I(p)

    stabit = 0

    while true
        stabit += 1
        unscaledG = randn(ranks[1], ranks[2], ranks[3], ranks[4], p)
        G .= rescaleten(unscaledG, gscale)
        randU2 = randn(dimvals[2], ranks[2])
        U2 .= svd(randU2).U

        randU3 = randn(dimvals[1], ranks[3])
        U3 .= svd(randU3).U

        randU4 = randn(dimvals[2], ranks[4])
        U4 .= svd(randU4).U

        hosvdA = ttensor(G, [U1, U2, U3, U4, Matrix(U5)])
        A .= full(hosvdA)
        varA = tenmat(A, row=[1, 2])
        if isstable(varA; maxeigen)
            break
        end
    end
    return (; A, G, U1, U2, U3, U4, U5, stabit)
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
    obs::Int;
    A=nothing,
    p::Int=1,
    snr=0.7,
)
    if isnothing(A)
        A = generatetuckercoef(dimvals, ranks, p)[1]
    end

    rho = spectralradius(makecompanion(tenmat(A, row=[1, 2])))
    Σ = diagm(repeat([rho / snr], dimvals[1] * dimvals[2]))
    d = MultivariateNormal(zeros(dimvals[1] * dimvals[2]), Σ)

    data = zeros(dimvals[1], dimvals[2], obs)
    for i = (p+1):obs
        vecϵ = rand(d)
        ϵ = reshape(vecϵ, (dimvals[1], dimvals[2]))

        data[:, :, i] .= contract(A[:, :, :, :, 1], [3, 4], data[:, :, i-1], [1, 2]) + ϵ

        for j = 2:p
            data[:, :, i] .+= contract(A[:, :, :, :, j], [3, 4], data[:, :, i-j], [1, 2])
        end
    end

    return (; data, A, Σ)
end
