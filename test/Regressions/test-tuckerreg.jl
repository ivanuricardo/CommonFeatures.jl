@testset "Tucker Est" begin
    using TensorToolbox
    Random.seed!(20231228)
    # Match Tucker regression with VAR
    dimvals = [4, 3]
    ranks = [4, 3, 4, 3]
    obs = 1000
    scale = 4
    p = 1
    snr = 0.7

    eta = 1e-04
    maxiter = 1000
    p = 1
    ϵ = 1e-02

    A, _ = generatetuckercoef(dimvals, ranks, p)
    marsim = simulatetuckerdata(dimvals, ranks, obs; A, p, snr)
    mardata = marsim.data
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(mardata)
    yy = tenmat(origy, col=3) .- mean(tenmat(origy, col=3), dims=2)
    xx = tenmat(lagy, col=4) .- mean(tenmat(lagy, col=4), dims=2)

    varest = yy * xx' * inv(xx * xx')

    tuckerest = tuckerreg(mardata, ranks; eta, maxiter, p, ϵ)
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
    snr = 0.7

    eta = 1e-04
    maxiter = 1000
    ϵ = 1e-02

    A, _ = generatetuckercoef(dimvals, ranks, p)
    marsim = simulatetuckerdata(dimvals, ranks, obs; A, p, snr)
    mardata = marsim.data
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(mardata, p)
    yy = tenmat(origy, col=3) .- mean(tenmat(origy, col=3), dims=2)
    xx = tenmat(lagy, col=4) .- mean(tenmat(lagy, col=4), dims=2)

    varest = yy * xx' * inv(xx * xx')

    tuckerest = tuckerreg(mardata, ranks; eta, maxiter, p, ϵ)
    flattuck = tenmat(tuckerest.A, row=[1, 2])
    @test varest ≈ flattuck
end

@testset "Tucker Est p = 3" begin
    using TensorToolbox
    Random.seed!(20231228)
    # Match Tucker regression with VAR
    dimvals = [4, 3]
    ranks = [4, 3, 4, 3]
    obs = 1000
    scale = 3
    p = 3
    snr = 0.7

    eta = 1e-04
    maxiter = 100
    ϵ = 1e-02

    A, _ = generatetuckercoef(dimvals, ranks, p)
    marsim = simulatetuckerdata(dimvals, ranks, obs; A, p, snr)
    mardata = marsim.data
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(mardata, p)
    yy = tenmat(origy, col=3) .- mean(tenmat(origy, col=3), dims=2)
    xx = tenmat(lagy, col=4) .- mean(tenmat(lagy, col=4), dims=2)

    varest = yy * xx' * inv(xx * xx')

    tuckerest = tuckerreg(mardata, ranks; eta, maxiter, p, ϵ)
    flattuck = tenmat(tuckerest.A, row=[1, 2])
    @test varest ≈ flattuck
end

@testset "Gradients p = 1" begin
    using TensorToolbox
    Random.seed!(20231228)

    dimvals = [6, 5]
    ranks = [5, 4, 3, 2]
    selectedranks = [5, 4, 3, 2]
    obs = 100
    scale = 5
    p = 1
    snr = 0.7

    eta = 1e-04
    maxiter = 10
    ϵ = 1e-02

    A, _ = generatetuckercoef(dimvals, ranks, p)
    marsim = simulatetuckerdata(dimvals, ranks, obs; A, p, snr)
    mardata = marsim.data

    tuck1 = tuckerreg(mardata, selectedranks; eta, maxiter, p, ϵ)
    tuck2 = tuckerreg2(mardata, selectedranks; eta, maxiter, p, ϵ)
    mask1 = .!isnan.(tuck1.fullgrads)
    mask2 = .!isnan.(tuck2.fullgrads)
    grad1 = tuck1.fullgrads[mask1]
    grad2 = tuck2.fullgrads[mask2]
    @test grad1 ≈ grad2
end

@testset "Gradients p = 2" begin
    using TensorToolbox
    Random.seed!(20231228)

    dimvals = [6, 5]
    ranks = [5, 4, 3, 2]
    selectedranks = [5, 4, 3, 2]
    obs = 100
    scale = 5
    p = 2
    snr = 0.7

    eta = 1e-04
    maxiter = 10
    ϵ = 1e-02

    A, _ = generatetuckercoef(dimvals, ranks, p)
    marsim = simulatetuckerdata(dimvals, ranks, obs; A, p, snr)
    mardata = marsim.data

    tuck1 = tuckerreg(mardata, selectedranks; eta, maxiter, p, ϵ)
    tuck2 = tuckerreg2(mardata, selectedranks; eta, maxiter, p, ϵ)
    mask1 = .!isnan.(tuck1.fullgrads)
    mask2 = .!isnan.(tuck2.fullgrads)
    grad1 = tuck1.fullgrads[mask1]
    grad2 = tuck2.fullgrads[mask2]
    @test grad1 ≈ grad2
end

@testset "Gradients p = 3" begin
    using TensorToolbox
    Random.seed!(20231228)

    dimvals = [6, 5]
    ranks = [5, 4, 3, 2]
    selectedranks = [5, 4, 3, 2]
    obs = 100
    scale = 5
    p = 3
    snr = 0.7

    eta = 1e-04
    maxiter = 10
    ϵ = 1e-02

    A, _ = generatetuckercoef(dimvals, ranks, p)
    marsim = simulatetuckerdata(dimvals, ranks, obs; A, p, snr)
    mardata = marsim.data

    tuck1 = tuckerreg(mardata, selectedranks; eta, maxiter, p, ϵ)
    tuck2 = tuckerreg2(mardata, selectedranks; eta, maxiter, p, ϵ)
    mask1 = .!isnan.(tuck1.fullgrads)
    mask2 = .!isnan.(tuck2.fullgrads)
    grad1 = tuck1.fullgrads[mask1]
    grad2 = tuck2.fullgrads[mask2]
    @test grad1 ≈ grad2
end
