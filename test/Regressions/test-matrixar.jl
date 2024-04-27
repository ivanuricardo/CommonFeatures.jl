@testset "Full RRVAR" begin
    using TensorToolbox
    Random.seed!(20231228)
    # Match Tucker regression with VAR
    N = 5
    r = 5
    obs = 1000
    p = 1
    maxeigen = 0.9

    C, _ = generaterrvarcoef(N, r, p, maxeigen=maxeigen)
    rrvarsim = simulaterrvardata(N, r, p; obs=obs, C=C)
    rrvardata = rrvarsim.data
    origy = vlag(rrvardata)[1:N, :]
    lagy = vlag(rrvardata)[(N+1):end, :]
    yy = origy .- mean(origy, dims=2)
    xx = lagy .- mean(lagy, dims=2)

    varest = yy * xx' * inv(xx * xx')

    rrvarest = rrvar(rrvardata, r, p)
    @test varest ≈ rrvarest.C
end

@testset "VAR estimation" begin
    N = 3
    p = 1
    burnin = 100
    obs = 5000 + burnin
    coefscale = 0.5
    snr = 0.7
    maxeigen = 0.9

    Random.seed!(20231228)
    A, _, _ = generatevarcoef(N, p; maxeigen, coefscale)

    repeats = 1000
    bias = zeros(repeats)
    for i in eachindex(bias)
        varsim = simulatevardata(N, p, obs; snr, coefscale, maxeigen, A, burnin)
        data = varsim.data
        origy = vlag(data, p)[1:N, :]
        lagy = vlag(data, p)[(N+1):end, :]
        yy = origy .- mean(origy, dims=2)
        xx = lagy .- mean(lagy, dims=2)

        varest = yy * xx' * inv(xx * xx')
        bias[i] = norm(varest - A)^2
    end


    @test mean(bias) < 0.1
end

@testset "MAR estimation p = 2" begin
    N = 12
    p = 2
    burnin = 100
    obs = 5000 + burnin
    coefscale = 0.1
    snr = 0.7
    maxeigen = 0.9

    Random.seed!(20231228)
    A, _, _ = generatevarcoef(N, p; maxeigen, coefscale)
    varsim = simulatevardata(N, p, obs; snr, coefscale, maxeigen, A, burnin)
    data = varsim.data
    origy = vlag(data, p)[1:N, :]
    lagy = vlag(data, p)[(N+1):end, :]
    yy = origy .- mean(origy, dims=2)
    xx = lagy .- mean(lagy, dims=2)

    varest = yy * xx' * inv(xx * xx')

    mardata = matten(data, [1, 2], [3], [4, 3, 5001])
    marest = art(mardata, p)

    @test tenmat(marest.tols, row=[1, 2]) ≈ varest
end
