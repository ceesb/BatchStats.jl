
using ProgressBars
using .Threads

function cor(A::AbstractMatrix, B::AbstractMatrix; 
                    dims = ndims(A), 
                    batchsize = 128)
    @assert dims == 2

    nA, ntraces = size(A)
    nB, ntraces2 = size(B)

    @assert ntraces == ntraces2

    cors = [BatchStats.BatchCorrelation(nA, nB, batchsize) for i in 1 : Threads.nthreads()]

    Threads.@threads for t in ProgressBar(1 : batchsize : ntraces)
        tid = Threads.threadid()
        e = min(ntraces, t - 1 + batchsize)
        s = @view(A[:, t : e])
        d = @view(B[:, t : e])
        add!(cors[tid], s, d)
    end

    reduce(add!, cors)
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

function meanvar(samples, data::AbstractMatrix{T}) where {T <: Integer}
    ntraces = size(samples, 2)
    nsamples = size(samples, 1)
    ndata = size(data, 1)
    nthreads = Threads.nthreads()

    vars = [Dict{T, BatchVariance}() for row = 1 : ndata, col = 1 : nthreads]

    Threads.@threads for t in ProgressBar(1 : ntraces)
        tid = Threads.threadid()
        s = @view(samples[:, t])
        for i in 1 : ndata
            v = data[i, t]
            get!(vars[i, tid], v, BatchVariance(nsamples))
            add!(vars[i, tid][v], s)
        end
    end

    combiner(x,y) = (add!(x, y); x)

    for i in 2 : nthreads
        for j in 1 : ndata
            mergewith(combiner, vars[j, 1], vars[j, i])
        end
    end

    result = vars[1 : ndata, 1]
    avals = [keys(r) |> collect |> sort for r in result]
    avarsvec = [[result[i][x] for x in vals] for (i, vals) in enumerate(avals)]

    return avals, avarsvec
end

function mergevars(f, vals, vars)
    k = f(first(vals))
    nsamples = getMean(first(vars)) |> length
    d = Dict{typeof(k), BatchVariance}()

    for (i, (val,var)) in enumerate(zip(vals, vars))
        fv = f(val)
        get!(d, fv, BatchVariance(nsamples))
        add!(d[fv], var)
    end

    vals = Base.keys(d) |> unique |> sort
    vars = [d[v] for v in vals]

    return vals, vars
end
