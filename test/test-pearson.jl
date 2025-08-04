using Test
using Statistics
using BatchStats

function testpearson()
    @testset "pearson stats" begin
    n = 1000
    nx = 3
    ny = 4
    x = rand(nx, n)
    y = rand(ny, n)

    ic = BatchCorrelation(nx, ny)
    for i in 1 : n
        @views add!(ic, x[:, i], y[:, i])
    end

    @test getMeanX(ic) ≈ mean(x; dims = 2)
    @test getMeanY(ic) ≈ mean(y; dims = 2)
    @test getVarianceX(ic) ≈ var(x; dims = 2)
    @test getVarianceY(ic) ≈ var(y; dims = 2)
    @test getCovariance(ic; corrected = true) ≈ cov(x, y; dims = 2, corrected = true)
    
    C = getCorrelation(ic)
    @test cor(x, y; dims = 2) ≈ C

    x2 = rand(nx, n)
    y2 = rand(ny, n)

    ic2 = BatchCorrelation(nx, ny)
    for i in 1 : n
        @views add!(ic2, x2[:, i], y2[:, i])
    end

    @test getMeanX(ic2) ≈ mean(x2; dims = 2)
    @test getMeanY(ic2) ≈ mean(y2; dims = 2)
    @test getVarianceX(ic2) ≈ var(x2; dims = 2)
    @test getVarianceY(ic2) ≈ var(y2; dims = 2)

    add!(ic, ic2)

    @test getMeanX(ic) ≈ mean(hcat(x,x2); dims = 2)
    @test getMeanY(ic) ≈ mean(hcat(y,y2); dims = 2)
    @test getVarianceX(ic) ≈ var(hcat(x,x2); dims = 2)
    @test getVarianceY(ic) ≈ var(hcat(y,y2); dims = 2)
    @test getCovariance(ic; corrected = true) ≈ cov(hcat(x,x2), hcat(y,y2); dims = 2, corrected = true)

    C2 = getCorrelation(ic)
    @test cor(hcat(x,x2), hcat(y,y2); dims = 2) ≈ C2

    x3 = rand(nx, n)
    y3 = rand(ny, n)

    batchsize = 16
    ic3 = BatchCorrelation(nx, ny, batchsize)
    for i in 1 : batchsize : n
        l = min(n, i - 1 + batchsize)
        @views add!(ic3, x3[:, i : l], y3[:, i : l])
    end

    @test getMeanX(ic3) ≈ mean(x3; dims = 2)
    @test getMeanY(ic3) ≈ mean(y3; dims = 2)
    @test getVarianceX(ic3) ≈ var(x3; dims = 2)
    @test getVarianceY(ic3) ≈ var(y3; dims = 2)
    add!(ic, ic3)

    @test getMeanX(ic) ≈ mean(hcat(x,x2,x3); dims = 2)
    @test getMeanY(ic) ≈ mean(hcat(y,y2,y3); dims = 2)
    @test getVarianceX(ic) ≈ var(hcat(x,x2,x3); dims = 2)
    @test getVarianceY(ic) ≈ var(hcat(y,y2,y3); dims = 2)
    @test getCovariance(ic; corrected = true) ≈ cov(hcat(x,x2,x3), hcat(y,y2,y3); dims = 2, corrected = true)

    C3 = getCorrelation(ic)
    @test cor(hcat(x,x2,x3), hcat(y,y2,y3); dims = 2) ≈ C3
    end
end
