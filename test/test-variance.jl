using Test
using Statistics
using BatchStats

function testvariance()
    @testset "variance stats" begin
    n = 1000
    nx = 3
    x = rand(nx, n)

    ic = BatchVariance(nx)
    for i in 1 : n
        @views add!(ic, x[:, i])
    end

    @test getMean(ic) ≈ mean(x; dims = 2)
    @test getVariance(ic) ≈ var(x; dims = 2)

    x2 = rand(nx, n)

    ic2 = BatchVariance(nx)
    for i in 1 : n
        @views add!(ic2, x2[:, i])
    end

    @test getMean(ic2) ≈ mean(x2; dims = 2)
    @test getVariance(ic2) ≈ var(x2; dims = 2)

    add!(ic, ic2)

    @test getMean(ic) ≈ mean(hcat(x,x2); dims = 2)
    @test getVariance(ic) ≈ var(hcat(x,x2); dims = 2)

    x3 = rand(nx, n)

    batchsize = 16
    ic3 = BatchVariance(nx)
    for i in 1 : batchsize : n
        l = min(n, i - 1 + batchsize)
        @views add!(ic3, x3[:, i : l])
    end

    @test getMean(ic3) ≈ mean(x3; dims = 2)
    @test getVariance(ic3) ≈ var(x3; dims = 2)

    add!(ic, ic3)

    @test getMean(ic) ≈ mean(hcat(x,x2,x3); dims = 2)
    @test getVariance(ic) ≈ var(hcat(x,x2,x3); dims = 2)
    end
end
