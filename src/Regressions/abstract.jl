abstract type CommonFeaturesModel end

struct LowRankTensorAutoRegression <: CommonFeaturesModel
    p::Int
    q::Int
end

struct ReducedRankAutoRegression <: CommonFeaturesModel
    C::AbstractMatrix
    B::AbstractMatrix
    A::AbstractMatrix
    leftnull::AbstractMatrix
    loglike::Float64
end

leftnull(Model <: ReducedRankAutoRegression) = Model.leftnull
