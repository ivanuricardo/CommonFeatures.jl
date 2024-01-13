"""
    tlag(ten_data::AbstractArray, p=1)

Create lagged arrays from a tensor-valued time series of the form N₁ × N₂ × … × Nₖ × T.

# Arguments:
- `ten_data`: A tensor-valued time series with dimensions N₁ × N₂ × … × Nₖ × T, where N₁, N₂, ..., Nₖ represent the spatial dimensions, and T represents the time dimension.
- `p`: The number of lags to be created. Default is 1, indicating a single lag.

# Returns:
- `original`: The original tensor without any lag.
- `lagged`: A tensor containing lagged versions of the original tensor, with dimensions N₁ × N₂ × … × Nₖ × (p * lastdim), where lastdim is the size of the last dimension of the input tensor.

# Example:
```julia
data = randn(3, 4, 5, 10)  # Example tensor with dimensions 3 × 4 × 5 × 10
original, lagged = tlag(data, 2)  # Create lagged arrays with 2 lags
```
"""
function tlag(ten_data::AbstractArray, p=1, pdims::Bool=false)
    N1, N2, obs = size(ten_data)

    fulllags = selectdim(ten_data, ndims(ten_data), (p+1):obs)
    for i in 1:p
        interlag = selectdim(ten_data, ndims(ten_data), (p-i+1):(obs-i))
        fulllags = cat(fulllags, interlag; dims=ndims(ten_data) - 1)
    end
    original = selectdim(fulllags, ndims(ten_data) - 1, 1:N2)
    lagged = selectdim(fulllags, ndims(ten_data) - 1, (N2+1):((p+1)*N2))
    if pdims
        lagged = reshape(lagged, (N1, N2, p, obs - p))
    end

    return original, lagged

end

"""
    ridgerankselect(est::AbstractArray, c::Real, r̄=collect(size(est)))::Vector{Int}

Used to provide an initial guess of the ranks for a Tucker regression using the framework of Lam and Yao (2011) and Wang et. al (2023).
Given a four-dimensional initial estimate `est` (typically from the art function), this function performs ridge rank selection on each mode of the tensor using a ridge parameter `c`.
The result is a vector of integers representing the selected ranks for each mode.

# Arguments
- `est::AbstractArray`: The input four-dimensional array (tensor) for ridge rank selection.
- `c::Real`: Ridge parameter controlling regularization strength.
- `r̄::Vector{Int} = collect(size(est))`: Vector specifying the maximum ranks to consider
  for each mode. Default is the size of the corresponding mode in `est`.

# Returns
- `Vector{Int}`: A vector containing the selected ranks for each mode of the input tensor.

# Example
```julia
est = rand(3, 4, 5, 6)
c = 0.1
selected_ranks = ridgerankselect(est, c)
```

# References

- Lam, Clifford, and Qiwei Yao. "Factor modeling for high-dimensional time series: inference for the number of factors." The Annals of Statistics (2012): 694-726.
- Wang, Di, et al. "High-dimensional vector autoregressive time series modeling via tensor decomposition." Journal of the American Statistical Association 117.539 (2022): 1338-1356.
"""
function ridgerankselect(est::AbstractArray, c::Real, r̄=collect(size(est)))::Vector{Int}
    S1 = svd(tenmat(est, 1)).S
    S2 = svd(tenmat(est, 2)).S
    S3 = svd(tenmat(est, 3)).S
    S4 = svd(tenmat(est, 4)).S

    r1 = findmax((S1[1:(r̄[1]-1)] .+ c) ./ (S1[2:r̄[1]] .+ c))[2]
    r2 = findmax((S2[1:(r̄[2]-1)] .+ c) ./ (S2[2:r̄[2]] .+ c))[2]
    r3 = findmax((S3[1:(r̄[3]-1)] .+ c) ./ (S3[2:r̄[3]] .+ c))[2]
    r4 = findmax((S4[1:(r̄[4]-1)] .+ c) ./ (S4[2:r̄[4]] .+ c))[2]

    return [r1, r2, r3, r4]
end

function rescaleten(A, target_norm::Real=5)
    current_norm = norm(A, 2)  # Compute the Frobenius norm of the matrix A

    # Rescale each element of the matrix
    scaled_matrix = A * (target_norm / current_norm)

    return scaled_matrix
end

