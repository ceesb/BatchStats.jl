using BenchmarkTools
using BatchStats
include(pkgdir(BatchStats) * "/test/lazydummyarray.jl")

nsamples = 1000
ntraces = 2000

function bench_batchvariance_vector(x, M)
    for i in 1 : ntraces
        add!(x, @view(M[:, i]))
    end
end

function do_bench_batchvariance_vector()
    b1 = @benchmark bench_batchvariance_vector(x, M) setup=(x = BatchVariance(nsamples); M = LazyDummyArray((nsamples, ntraces), 0x42))
    return b1
end

function do_batchvariance_vector()
    x = BatchVariance(nsamples)
    M = LazyDummyArray((nsamples, ntraces), 0x42)
    bench_batchvariance_vector(x, M)
end

batchsize = 100

function bench_batchvariance_matrix(x, M)
    for i in 1 : batchsize : ntraces
        add!(x, @view(M[:, i]))
    end
end

function do_bench_batchvariance_matrix()
    b1 = @benchmark bench_batchvariance_matrix(x, M) setup=(x = BatchVariance(nsamples); M = LazyDummyArray((nsamples, ntraces), 0x42))
    return b1
end

function do_batchvariance_matrix()
    x = BatchVariance(nsamples)
    M = LazyDummyArray((nsamples, ntraces), 0x42)
    bench_batchvariance_matrix(x, M)
end


function bench_meanvar(M, labels)
    BatchStats.meanvar(M, labels; batchsize = batchsize)
end

function do_bench_meanvar()
    b1 = @benchmark bench_meanvar(M, labels) setup=(M = LazyDummyArray((nsamples, ntraces), 0x42); labels = rand(0x00:0x01, 1, ntraces))
    return b1
end

function do_meanvar()
    M = LazyDummyArray((nsamples, ntraces), 0x42)
    labels = rand(0x00:0x01, 1, ntraces)
    bench_meanvar(M, labels)
end

