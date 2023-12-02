"""
    tensorols(Y, X)

Compute tensor ordinary least squares equation by equation.

# Arguments
- `Y::AbstractArray`: Tensor of size (n1, n2, ..., nk, T).
- `X::AbstractArray`: Tensor of size (p1, p2, ..., pk, T).

# Returns
- `result::AbstractArray`: Tensor of size (n1, n2, ..., nk, p1, p2, ..., pk).
"""
function tensorols(Y::AbstractArray, X::AbstractArray)
    fulldims = (size(Y)[1:end-1]..., size(X)[1:end-1]...)

    flaty = tenmat(Y, col=ndims(Y))
    flatx = tenmat(X, col=ndims(X))

    flatols = flaty * flatx' * inv(flatx * flatx')

    return reshape(flatols, fulldims)
end

