@testset "Tucker Data" begin
    Random.seed!(20231228)
    # Verify that Tucker data indeed generates a VAR model
    # We do this for one lag, then for two lags
    dimvals = [4, 3]
    ranks = [4, 3, 4, 3]
    obs = 1000
    scale = 4
    p = 1
    snr = 0.9

    marsim = simulatetuckerdata(dimvals, ranks, obs, nothing, p, snr)
    mardata = marsim.data
    vardata = tenmat(mardata, col=3)
    origy, lagy = tlag(mardata)
    yy = tenmat(origy, col=3) .- mean(tenmat(origy, col=3), dims=2)
    xx = tenmat(lagy, col=4) .- mean(tenmat(lagy, col=4), dims=2)

    varest = yy * xx' * inv(xx * xx')

    @test isapprox(tenmat(marsim.A, row=[1, 2]), varest, rtol=0.1)
end
