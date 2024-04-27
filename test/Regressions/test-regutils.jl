@testset "idhosvd" begin
    using TensorToolbox
    Random.seed!(20231228)

    X = randn(4, 3, 4, 3)
    r = [1, 1, 1, 1]
    idh = idhosvd(X, r)
    nhos = hosvd(X; reqrank=r)
    U1, U2, U3, U4 = idh.fmat
    nU1, nU2, nU3, nU4 = nhos.fmat

    @test U1 ≈ nU1
    @test U2 ≈ nU2
    @test U3 ≈ nU3
    @test U4 ≈ nU4
end

@testset "rescale tensor" begin
    Random.seed!(20231228)

    X = randn(4, 3, 2)
    scale = 4
    @test norm(rescaleten(X, scale)) ≈ scale
end
