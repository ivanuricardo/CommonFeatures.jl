@testset "Tucker Parameters" begin
    # The number of parameters minus the degrees of freedom
    # Although Wang et. al defines the degrees of freedom as the number of parameters, our degrees of freedom are the number of restrictions (orthogonality, sign restrictions, etc.)
    #
    # First test the example in Wang et. al

    k = 3 # d in the paper
    N1 = 20
    N2 = 20
    N3 = 20
    ranks = fill(2, 6)
    P = 1 # lags

    pars = tuckerpar([N1, N2, N3], ranks, P)

    @test pars == 280
end

@testset "Tucker Parameters p = 2" begin

    dimvals = [4, 3]
    ranks = [4, 3, 4, 3]
    p = 2

    pars = tuckerpar(dimvals, ranks, p)
    truepars = 12 * 12 * p

    @test pars == truepars
end

@testset "Tucker Condition" begin

    r1 = [4, 3, 4, 2]
    r2 = [1, 1, 1, 2]
    r3 = [1, 3, 3, 4]
    cond1 = CommonFeatures.tuckercondition(r1)
    cond2 = CommonFeatures.tuckercondition(r2)
    cond3 = CommonFeatures.tuckercondition(r3)


    @test cond1
    @test cond2 == false
    @test cond3
end
