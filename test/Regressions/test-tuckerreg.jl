@testset "Tucker Est" begin
    using TensorToolbox
    Random.seed!(20231228)
    # Match Tucker regression with VAR
    dimvals = [4, 3]
    ranks = [4, 3, 4, 3]
    obs = 1000
    scale = 4
    p = 1
    maxeigen = 0.9

    eta = 1e-04
    a = 1
    b = 1
    miniters = 100
    maxiters = 1000
    p = 1
    ϵ = 1e-02

    marsim = simulatetuckerdata(dimvals, ranks, obs, scale, p, maxeigen)
    mardata = marsim.tuckerdata
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(vardata)

    varest = origy * lagy' * inv(lagy * lagy')

    tuckerest = tuckerreg(mardata, ranks, eta, a, b, miniters, maxiters, p, ϵ)
    flattuck = tenmat(tuckerest.A, row=[1, 2])
    @test varest ≈ flattuck
end

@testset "Tucker Est p = 2" begin
    using TensorToolbox
    Random.seed!(20231228)
    # Match Tucker regression with VAR
    dimvals = [4, 3]
    ranks = [4, 3, 4, 6]
    obs = 1000
    scale = 3
    p = 2
    maxeigen = 0.9

    eta = 1e-04
    a = 1
    b = 1
    miniters = 100
    maxiters = 1000
    ϵ = 1e-02

    marsim = simulatetuckerdata(dimvals, ranks, obs, scale, p, maxeigen)
    mardata = marsim.tuckerdata
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(vardata, p)

    varest = origy * lagy' * inv(lagy * lagy')

    tuckerest = tuckerreg(mardata, ranks, eta, a, b, maxiters, p, ϵ)
    flattuck = tenmat(tuckerest.A, row=[1, 2])
    @test varest ≈ flattuck
end
