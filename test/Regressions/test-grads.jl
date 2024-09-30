
Random.seed!(20240921)
N1 = 4
N2 = 3
r1 = 2
r2 = 1
obs = 100

U1, U2, U3, U4, ϕ1, ϕ2 = generatemecmparams([N1, N2], [r1, r2], genphi=true)
ΔY = randn(N1, N2, obs)
Y = randn(N1, N2, obs)
mdy = reshape(ΔY, N1 * N2, obs)
my = reshape(Y, N1 * N2, obs)
D = 0.1 .* randn(N1, N2)
Σ1 = rand(Wishart(N1, diagm(ones(N1))))
Σ2 = rand(Wishart(N2, diagm(ones(N2))))

@testset "loss equivalence" begin
    losswithvec = objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)
    losswithmat = matobj(Y, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2)
    @test isapprox(losswithvec, losswithmat, atol=1e-8)
end

# Test gradients: We expect the differences between computed and true gradients to be close to zero
@testset "Gradient Tests" begin
    # Gradients for D
    truegradD = gradient(x -> objmecm(my, x, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), D)[1]
    approx_gradD = mecmsumres(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegradD, approx_gradD, atol=1e-8)

    # Gradients for U1
    truegrad1 = gradient(x -> objmecm(my, D, x, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U1)[1]
    approx_grad1 = U1grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegrad1, approx_grad1, atol=1e-8)

    # Gradients for U2
    truegrad2 = gradient(x -> objmecm(my, D, U1, x, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U2)[1]
    approx_grad2 = U2grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegrad2, approx_grad2, atol=1e-8)

    # Gradients for U3
    truegrad3 = gradient(x -> objmecm(my, D, U1, U2, x, U4, Σ1, Σ2, ϕ1, ϕ2), U3)[1]
    approx_grad3 = U3grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegrad3, approx_grad3, atol=1e-8)

    # Gradients for U4
    truegrad4 = gradient(x -> objmecm(my, D, U1, U2, U3, x, Σ1, Σ2, ϕ1, ϕ2), U4)[1]
    approx_grad4 = U4grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegrad4, approx_grad4, atol=1e-8)

    # Gradients for ϕ1
    truegradϕ1 = gradient(x -> objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, x, ϕ2), ϕ1)[1]
    approx_gradϕ1 = ϕ1grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegradϕ1, approx_gradϕ1, atol=1e-8)

    # Gradients for ϕ2
    truegradϕ2 = gradient(x -> objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, x), ϕ2)[1]
    approx_gradϕ2 = ϕ2grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegradϕ2, approx_gradϕ2, atol=1e-8)

    truegradΣ1 = gradient(x -> objmecm(my, D, U1, U2, U3, U4, x, Σ2, ϕ1, ϕ2), Σ1)[1]
    approx_gradΣ1 = Σ1grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegradΣ1, approx_gradΣ1, rtol=1e-8)

    truegradΣ2 = gradient(x -> objmecm(my, D, U1, U2, U3, U4, Σ1, x, ϕ1, ϕ2), Σ2)[1]
    approx_gradΣ2 = Σ2grad(Y, U1, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2, D)
    @test isapprox(truegradΣ2, approx_gradΣ2, rtol=1e-8)
end

# Test Hessians: We expect the Hessians to be close to equal
@testset "Hessian Tests" begin
    # Hessian for U1
    hess1 = hessian(x -> objmecm(my, D, x, U2, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U1)
    approx_hess1 = U1hessian(Y, U2, U3, U4, Σ1, Σ2)
    @test isapprox(maximum(abs.(eigvals(hess1))), maximum(abs.(eigvals(approx_hess1))), atol=1e-8)

    # Hessian for U2
    hess2 = hessian(x -> objmecm(my, D, U1, x, U3, U4, Σ1, Σ2, ϕ1, ϕ2), U2)
    approx_hess2 = U2hessian(Y, U1, U3, U4, Σ1, Σ2)
    @test isapprox(maximum(abs.(eigvals(hess2))), maximum(abs.(eigvals(approx_hess2))), atol=1e-8)

    # Hessian for U3README
    hess3 = hessian(x -> objmecm(my, D, U1, U2, x, U4, Σ1, Σ2, ϕ1, ϕ2), U3)
    approx_hess3 = U3hessian(Y, U1, U2, U4, Σ1, Σ2)
    @test isapprox(maximum(abs.(eigvals(hess3))), maximum(abs.(eigvals(approx_hess3))), atol=1e-8)

    # Hessian for U4
    hess4 = hessian(x -> objmecm(my, D, U1, U2, U3, x, Σ1, Σ2, ϕ1, ϕ2), U4)
    approx_hess4 = U4hessian(Y, U1, U2, U3, Σ1, Σ2)
    @test isapprox(maximum(abs.(eigvals(hess4))), maximum(abs.(eigvals(approx_hess4))), atol=1e-8)

    # Hessian for ϕ1
    hessϕ1 = hessian(x -> objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, x, ϕ2), ϕ1)
    approx_hessϕ1 = ϕ1hessian(Y, ϕ2, Σ1, Σ2)
    @test isapprox(maximum(abs.(eigvals(hessϕ1))), maximum(abs.(eigvals(approx_hessϕ1))), rtol=1e-8)

    # Hessian for ϕ2
    hessϕ2 = hessian(x -> objmecm(my, D, U1, U2, U3, U4, Σ1, Σ2, ϕ1, x), ϕ2)
    approx_hessϕ2 = ϕ2hessian(Y, ϕ1, Σ1, Σ2)
    @test isapprox(maximum(abs.(eigvals(hessϕ2))), maximum(abs.(eigvals(approx_hessϕ2))), rtol=1e-8)
end
