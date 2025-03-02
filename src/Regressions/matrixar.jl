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

    flaty = tenmat(Y, col = ndims(Y))
    flatx = tenmat(X, col = ndims(X))

    flatols = flaty * flatx' / (flatx * flatx')

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
function art(Y::AbstractArray, p::Int = 1; center::Bool = true, stdize::Bool = false)
    origy, laggedy = tlag(Y, p)
    if stdize
        # Only standardize the lagged values and not the original!
        stdlagy = (laggedy ./ std(laggedy, dims = 4))
        cenlag = stdlagy .- mean(stdlagy, dims = 4)
    elseif center
        cenorig = origy .- mean(origy, dims = 3)
        cenlag = laggedy .- mean(laggedy, dims = 4)
    end
    tols = tensorols(cenorig, cenlag)

    return (tols = tols, cenorig = cenorig, cenlag = cenlag)
end

function var_coef(data, p; stdize = false)
    k, _ = size(data)

    origy = vlag(data, p)[1:k, :]
    laggedy = vlag(data, p)[(k+1):end, :]
    if stdize
        stdorigy = (origy ./ std(origy, dims = 2))
        cenorig = stdorigy .- mean(stdorigy, dims = 2)
        stdlagy = (laggedy ./ std(laggedy, dims = 2))
        cenlag = stdlagy .- mean(stdlagy, dims = 2)
    else
        cenorig = origy .- mean(origy, dims = 2)
        cenlag = laggedy .- mean(laggedy, dims = 2)
    end
    coef = cenorig * cenlag' * inv(cenlag * cenlag')

    return coef
end

"""
    rrvar(Y::AbstractMatrix, r::Int, p::Int)

Compute reduced-rank vector autoregressive (RRVAR) model parameters.

# Arguments
- `Y::AbstractMatrix`: Input matrix with dimensions (k, T), where k is the number of variables and T is the number of observations.
- `r::Int`: Rank of the reduced-rank model.
- `p::Int`: Order of the autoregressive model.

# Returns
- `C::Matrix`: Reduced-rank VAR coefficient matrix.
- `A::Matrix`: Left reduced-rank matrix.
- `B::Matrix`: Right reduced-rank matrix.

# Details
This function computes the reduced-rank VAR parameters using the method.

"""
function rrvar(vardata::AbstractMatrix, r::Int, p::Int; stdize::Bool = false)
    k, obs = size(vardata)

    origy = vlag(vardata, p)[1:k, :]
    laggedy = vlag(vardata, p)[(k+1):end, :]
    if stdize
        stdorigy = (origy ./ std(origy, dims = 2))
        cenorig = stdorigy .- mean(stdorigy, dims = 2)
        stdlagy = (laggedy ./ std(laggedy, dims = 2))
        cenlag = stdlagy .- mean(stdlagy, dims = 2)
    else
        cenorig = origy .- mean(origy, dims = 2)
        cenlag = laggedy .- mean(laggedy, dims = 2)
    end

    # Compute covariance matrices
    cov_x = cov(cenlag')
    cov_yx = cov(cenorig', cenlag')
    cov_xy = cov_yx'

    # Compute weighted matrix
    weighted_matrix = cov_yx * inv(cov_x) * cov_xy

    # Compute eigen decomposition
    eigen_weighted = eigen(weighted_matrix, permute = false)
    prevecs = eigen_weighted.vectors[:, end:-1:1]

    # Select r eigenvectors
    Vt = prevecs[:, 1:r]

    # Compute RRVAR parameters
    A = Vt
    B = Vt' * cov_yx / (cov_x)
    C = A * B

    rrvarerr = cenorig - C * cenlag
    loglike = logdet(rrvarerr * rrvarerr' / (obs - p))
    leftnullA = CommonFeatures.leftnull(A)

    return ReducedRankAutoRegression(C, B, A, leftnullA, loglike)
end
