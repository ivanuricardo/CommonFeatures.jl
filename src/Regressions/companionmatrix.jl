@doc """
    makecompanion(B::AbstractMatrix{T}) where {T}
    
Create the VAR companion matrix.

Given a VAR of the form

```math
y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + \varepsilon_t
```

The companion matrix is given by 

```math
C = \begin{bmatrix}
    B_1 & B_2 & \dots & B_{p-1} & B_p \\
    I_n & O & \dots & O & O \\
    O & I_n & & O & O \\
    \vdots & & \ddots & \vdots & \vdots \\ 
    O & O & \dots & I_n & O
\end{bmatrix}
```

Thus, ``B`` is a ``np\times np`` matrix. 

## Arguments

-`B::AbstractMatrix{T}`: Lag matrix in the form required for a `VAR` model. See
    the documentation of `VAR`.


## References

- Kilian, L., & Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis:
  (1st ed.). Cambridge University Press. https://doi.org/10.1017/9781108164818


"""
function makecompanion(B::AbstractMatrix{T}) where {T}
    n = Int(size(B, 1))
    p = Int(size(B, 2) / n)
    ident = diagm(fill(T(1), n * (p - 1)))
    companion_lower = hcat(ident, zeros(n * (p - 1), n))
    companion = vcat(B, companion_lower)
    return companion
end

@doc """
    isstable(var)

Check the stability of a VAR (Vector Autoregressive) model.

This function checks the stability of a VAR model by analyzing its companion matrix eigenvalues.
A VAR model is considered stable if all the eigenvalues of its companion matrix are within the unit circle.

## Arguments

- `var`: Lag matrix in the form required for a `VAR` model. See the documentation of `VAR`.

## Returns

- `Bool`: Returns `true` if the VAR model is stable, and `false` otherwise.

## Example

```julia
B = [1.0 2.0;
     3.0 4.0]
var_stable = isstable(B)  # Returns true or false based on the stability of the VAR model
```
## Note
The stability of a VAR model is determined by analyzing the eigenvalues of its companion matrix.
The companion matrix is constructed using the makecompanion function.

## See Also
- `makecompanion`: Function to create the VAR companion matrix.
"""
function isstable(var)
    C = makecompanion(var)
    return maximum(abs.(eigen(C).values)) < 1.0
end
