import Base.show

abstract type CommonFeaturesModel end

struct LowRankTensorAutoRegression <: CommonFeaturesModel
    G::AbstractArray
    U::AbstractVector{<:AbstractArray}
    A::AbstractArray
    iters::Int
    fullgrads::AbstractMatrix
    residuals::AbstractMatrix
    converged::Bool
end

struct ReducedRankAutoRegression <: CommonFeaturesModel
    C::AbstractMatrix
    B::AbstractVecOrMat
    A::AbstractVecOrMat
    leftnull::AbstractVector
    loglike::Float64
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::ReducedRankAutoRegression)
    summary(io, F)
    println(io, "\nA factor:")
    show(io, mime, F.A)
    println(io, "\nB factor:")
    show(io, mime, F.B)
    println(io, "\nC coef:")
    show(io, mime, F.C)
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::LowRankTensorAutoRegression)
    summary(io, F)
    println(io, "\nG core:")
    show(io, mime, F.G)
    println(io, "\nU factors:")
    show(io, mime, F.U)
    println(io, "\nA coef:")
    show(io, mime, F.A)
end
