import Statistics
import BatchStats
using Test

function testvar()
    @testset "var wrapper" begin
    nx = 100
    nobservations = 1000
    X = rand(nx,nobservations)
    result = BatchStats.var(X; dims = 2) 
    v = BatchStats.getVariance(result)
    m = BatchStats.getMean(result)
    @test v ≈ Statistics.var(X; dims = 2)
    @test m ≈ Statistics.mean(X; dims = 2)
    end
end

function testcor()
    @testset "cor wrapper" begin
    nx = 100
    ny = 33
    nobservations = 1000
    X = rand(nx,nobservations)
    Y = rand(ny,nobservations)
    result = BatchStats.cor(X, Y; dims = 2) 
    c = BatchStats.getCorrelation(result)
    @test c ≈ Statistics.cor(X,Y; dims = 2)
    end
end