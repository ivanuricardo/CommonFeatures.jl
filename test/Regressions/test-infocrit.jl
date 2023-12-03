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
