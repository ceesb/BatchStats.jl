
using ProgressBars
using .Threads

function cor(A::AbstractMatrix, B::AbstractMatrix; 
                dims = ndims(A), 
                progress = false, 
                batchsize = 128)
    @assert dims == ndims(A)
    nslicesA = prod(size(A, d) for d in dims)
    nslicesB = prod(size(B, d) for d in dims)
    nslicesA == nslicesB || error(
        "number of slices in A $nslicesA not equal to slices in B $nslicesB over dimension $dims")
    nslices = nslicesA

    nelems_per_sliceA = div(length(A), nslices)
    nelems_per_sliceB = div(length(B), nslices)

    if progress
        # bar = Progress(nslices)
        bar = ProgressBar(1:nslices)
    end

    chunk_size = cld(nslices, nthreads())

    tasks = [@spawn begin
        e = min(j + chunk_size - 1, nslices)
        m = BatchCorrelation(
                        nelems_per_sliceA, 
                        nelems_per_sliceB,
                        batchsize)

        for i in j : batchsize : e
            l = min(i - 1 + batchsize, e)
            add!(m, 
                @view(A[:, i : l]), 
                @view(B[:, i : l]))

            if progress
                # next!(bar; step = nloops)
                update(bar, l - i + 1)
            end
        end

        m
    end for j in 1 : chunk_size : nslices]

    reduce(add!, fetch.(tasks))
end

function var(A::AbstractMatrix; 
                dims = ndims(A), 
                progress = false, 
                batchsize = 128)
    dims == ndims(A) || 
        error("only reduction in dims = 2 is recommended because of batch performance, call BatchStats.var(A') if you really want this")
    nslices = prod(size(A, d) for d in dims)
    nelems_per_slice = div(length(A), nslices)

    if progress
        # bar = Progress(nslices)
        bar = ProgressBar(1:nslices)
    end

    chunk_size = cld(nslices, nthreads())

    tasks = [@spawn begin
        e = min(j + chunk_size - 1, nslices)
        m = BatchVariance(
                        nelems_per_slice)
        
        for i in j : batchsize : e
            l = min(i - 1 + batchsize, e)
            add!(m, 
                @view(A[:, i : l]), 
            )

            if progress
                # next!(bar; step = nloops)
                update(bar, nloops)
            end
        end

        m
    end for j in 1 : chunk_size : nslices]

    reduce(add!, fetch.(tasks))
end
