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
function tlag(ten_data::AbstractArray, p=1)

    lastdim = size(ten_data)[ndims(ten_data)-1]
    obs = size(ten_data)[ndims(ten_data)]
    fulllags = selectdim(ten_data, ndims(ten_data), (p+1):obs)
    for i in 1:p
        interlag = selectdim(ten_data, ndims(ten_data), (p-i+1):(obs-i))
        fulllags = cat(fulllags, interlag; dims=ndims(ten_data) - 1)
    end
    original = selectdim(fulllags, ndims(ten_data) - 1, 1:lastdim)
    lagged = selectdim(fulllags, ndims(ten_data) - 1, (lastdim+1):((p+1)*lastdim))

    return original, lagged

end
