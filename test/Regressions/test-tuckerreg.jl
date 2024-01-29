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

    marsim = simulatetuckerdata(dimvals, ranks, obs, nothing, scale, p, maxeigen)
    mardata = marsim.data
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(mardata)
    yy = tenmat(origy, col=3)
    xx = tenmat(lagy, col=3)

    varest = yy * xx' * inv(xx * xx')

    tuckerest = tuckerreg(mardata, ranks, eta, maxiters, p, ϵ)
    flattuck = tenmat(tuckerest.A, row=[1, 2])
    @test varest ≈ flattuck
end

@testset "Tucker Est p = 2" begin
    using TensorToolbox
    Random.seed!(20231228)
    # Match Tucker regression with VAR
    dimvals = [4, 3]
    ranks = [4, 3, 4, 3]
    obs = 1000
    scale = 3
    p = 2
    maxeigen = 0.9

    eta = 1e-04
    a = 1
    b = 1
    maxiters = 1000
    ϵ = 1e-02

    marsim = simulatetuckerdata(dimvals, ranks, obs, nothing, scale, p, maxeigen)
    mardata = marsim.data
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(mardata, p)
    yy = tenmat(origy, col=3)
    xx = tenmat(lagy, col=3)

    varest = yy * xx' * inv(xx * xx')

    tuckerest = tuckerreg(mardata, ranks, eta, maxiters, p, ϵ)
    flattuck = tenmat(tuckerest.A, row=[1, 2])
    @test varest ≈ flattuck
end

@testset "Gradients p = 1" begin
    using TensorToolbox
    Random.seed!(20231228)

    dimvals = [4, 3]
    ranks = [3, 2, 3, 2]
    selectedranks = [2, 1, 2, 1]
    obs = 100
    scale = 5
    p = 1
    maxeigen = 0.9

    eta = 1e-04
    a = 1
    b = 1
    maxiters = 100
    ϵ = 1e-02

    marsim = simulatetuckerdata(dimvals, ranks, obs, nothing, scale, p, maxeigen)
    mardata = marsim.data

    tuck1 = tuckerreg(mardata, selectedranks, eta, maxiters, p, ϵ)
    tuck2 = tuckerreg2(mardata, selectedranks, eta, maxiters, p, ϵ)
    @test tuck1.fullgrads ≈ tuck2.fullgrads
end

@testset "Gradients p = 2" begin
    using TensorToolbox
    Random.seed!(20231228)

    dimvals = [4, 3]
    ranks = [3, 2, 3, 2]
    selectedranks = [2, 1, 2, 1]
    obs = 100
    scale = 5
    p = 2
    maxeigen = 0.9

    eta = 1e-04
    a = 1
    b = 1
    maxiters = 100
    ϵ = 1e-02

    marsim = simulatetuckerdata(dimvals, ranks, obs, nothing, scale, p, maxeigen)
    mardata = marsim.data

    tuck1 = tuckerreg(mardata, selectedranks, eta, maxiters, p, ϵ)
    tuck2 = tuckerreg2(mardata, selectedranks, eta, maxiters, p, ϵ)
    @test tuck1.fullgrads ≈ tuck2.fullgrads
end
