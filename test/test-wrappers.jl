import Statistics
import BatchStats
using Test

function testvar()
    nx = 100
    nobservations = 1000
    X = rand(nx,nobservations)
    result = BatchStats.var(X; dims = 2) 
    v = BatchStats.getVariance(result)
    m = BatchStats.getMean(result)
    @test v ≈ Statistics.var(X; dims = 2)
    @test m ≈ Statistics.mean(X; dims = 2)
end

function testcor(progress = true)
    nx = 100
    ny = 33
    nobservations = 2000
    X = rand(nx,nobservations)
    Y = rand(ny,nobservations)
    result = BatchStats.cor(X, Y; dims = 2, progress = progress) 
    c = BatchStats.getCorrelation(result)
    @test c ≈ Statistics.cor(X,Y; dims = 2)
end

function testmeanvar()
    nx = 100
    nobservations = 1000
    ndata = 2
    X = rand(nx, nobservations)
    labels = rand(0:7, ndata, nobservations)

    meanvars = [BatchStats.BatchVariance(nx) for row = 1 : ndata, col = 1 : 8]
    for (col,label) in enumerate(eachcol(labels))
        for (row,data) in enumerate(label)
            BatchStats.add!(meanvars[row, data + 1], @view(X[:, col]))
        end
    end
    
    vals, meanvars_ = BatchStats.meanvar(X, labels)

    @test vals[1] == 0 : 7
    @test vals[2] == 0 : 7

    for i = 1 : ndata
        for j = 1 : 8
            @test BatchStats.getMean(meanvars[i, j]) ≈ BatchStats.getMean(meanvars_[i][j])
            @test BatchStats.getVariance(meanvars[i, j]) ≈ BatchStats.getVariance(meanvars_[i][j])
        end
    end

    labels8 = UInt8.(labels)
    vals8, meanvars8_ = BatchStats.meanvar(X, labels8)

    @test vals8[1] == 0 : 7
    @test vals8[2] == 0 : 7

    for i = 1 : ndata
        for j = 1 : 8
            @test BatchStats.getMean(meanvars[i, j]) ≈ BatchStats.getMean(meanvars8_[i][j])
            @test BatchStats.getVariance(meanvars[i, j]) ≈ BatchStats.getVariance(meanvars8_[i][j])
        end
    end

end



