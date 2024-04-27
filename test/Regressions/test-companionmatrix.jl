@testset "Companion Matrix" begin

    Random.seed!(1234)
    B = randn(4, 8)
    makecomp = makecompanion(B)
    @test makecomp[5:8, 1:4] == I
    @test makecomp[5:8, 5:8] == zeros(4, 4)
end

@testset "Stability" begin

    Random.seed!(1234)

    scale = 0.1
    B = scale * randn(4, 8)
    makecomp = makecompanion(B)

    @test isstable(makecomp; maxeigen=1.0) == true

    C = [1 2; 1 2]
    @test spectralradius(C) ≈ 3.0
    @test isstable(C; maxeigen=1.0) == false


end
