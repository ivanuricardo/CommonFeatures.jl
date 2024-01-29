@testset "Tucker Data" begin
    Random.seed!(20231228)
    # Verify that Tucker data indeed generates a VAR model
    # We do this for one lag, then for two lags
    dimvals = [4, 3]
    ranks = [4, 3, 4, 3]
    obs = 4000
    scale = 4
    p = 1
    maxeigen = 0.9

    marsim = simulatetuckerdata(dimvals, ranks, obs, nothing, scale, p, maxeigen)
    mardata = marsim.data
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(mardata)
    yy = tenmat(origy, col=3)
    xx = tenmat(lagy, col=3)

    varest = yy * xx' * inv(xx * xx')

    @test isapprox(tenmat(marsim.A, row=[1, 2]), varest, rtol=0.1)
end
