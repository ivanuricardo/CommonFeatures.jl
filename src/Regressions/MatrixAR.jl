"""
    tensorols(Y, X)

Compute tensor ordinary least squares equation by equation.

# Arguments
- `Y::AbstractArray`: Tensor of size (n1, n2, ..., nk, T).
- `X::AbstractArray`: Tensor of size (p1, p2, ..., pk, T).

# Returns
- `result`: Tensor of size (n1, n2, ..., nk, p1, p2, ..., pk).
"""
function tensorols(Y::AbstractArray, X::AbstractArray)
    fulldims = (size(Y)[1:end-1]..., size(X)[1:end-1]...)

    flaty = tenmat(Y, col=ndims(Y))
    flatx = tenmat(X, col=ndims(X))

    flatols = flaty * flatx' * inv(flatx * flatx')

    return reshape(flatols, fulldims)
end

"""
    art(Y, p)

Compute the Tensor Autoregressive (ART) model using tensor ordinary least squares.

# Arguments
- `Y::AbstractArray`: Matrix or Tensor of size (n1, n2, T) or (n1, n2, ..., nk, T).
- `p::Int`: Order of the autoregressive model.

# Returns
- `result`: Tensor of size (n1, n2, ..., nk, n1, ..., nk).

# Examples
```julia
Y = rand(3, 2, 100)
p = 2

result = art(Y, p)
```
"""
function art(Y::AbstractArray, p::Int=1)
    origy, laggedy = tlag(Y, p)
    tols = tensorols(origy, laggedy)

    return tols
end
