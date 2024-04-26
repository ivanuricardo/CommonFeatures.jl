@testset "Full RRVAR" begin
    using TensorToolbox
    Random.seed!(20231228)
    # Match Tucker regression with VAR
    N = 5
    r = 5
    obs = 1000
    p = 1
    maxeigen = 0.9

    rrvarsim = simulaterrvardata(N, r, obs, nothing, p, maxeigen)
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
    maxeigen = 0.9

    Random.seed!(20231228)
    A, _, _ = generatevarcoef(N, p, maxeigen, coefscale)

    repeats = 1000
    bias = zeros(repeats)
    for i in eachindex(bias)
        varsim = simulatevardata(N, p, obs, coefscale, maxeigen, A, burnin)
        data = varsim.data
        origy = vlag(data)[1:N, :]
        lagy = vlag(data)[(N+1):end, :]
        yy = origy .- mean(origy, dims=2)
        xx = lagy .- mean(lagy, dims=2)

        varest = yy * xx' * inv(xx * xx')
        bias[i] = norm(varest - A)^2
    end


    @test mean(bias) < 0.1
end
