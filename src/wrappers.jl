
using Base.Threads
using ProgressMeter

function cor(A::AbstractMatrix, B::AbstractMatrix;
             dims = ndims(A),
             batchsize::Int = 128,
             progress::Bool = true)
    @assert dims == 2
    nA, ntraces = size(A)
    nB, ntraces2 = size(B)
    @assert ntraces == ntraces2

    nslices = ntraces
    ntasks = nthreads()
    chunk_size = div(nslices, ntasks, RoundUp)
    pbar = Progress(nslices; enabled = progress, showspeed = true)
    tasks = Vector{Task}(undef, ntasks)

    for task_idx in 1:ntasks
        j = (task_idx - 1)*chunk_size + 1
        tasks[task_idx] = @spawn begin
            m = BatchStats.BatchCorrelation(nA, nB, batchsize)
            e = min(j + chunk_size - 1, nslices)
            for i in j : batchsize : e
                l = min(i - 1 + batchsize, e)
                tr = i:l
                add!(m, @view(A[:, tr]), @view(B[:, tr]))
                next!(pbar; step = length(tr))
            end
            m
        end
    end

    ret = reduce(add!, fetch.(tasks))

    finish!(pbar)

    return ret
end


function var(A::AbstractMatrix; 
                dims = ndims(A),
                batchsize = 128,
                progress = true)
    @assert dims == 2
    nA = size(A, 1)
    nslices = size(A, 2)
    ntasks = nthreads()
    chunk_size = div(nslices, ntasks, RoundUp)
    pbar = Progress(nslices; enabled = progress, showspeed = true)
    tasks = Vector{Task}(undef, ntasks)

    tasks = [@spawn begin
        e = min(j + chunk_size - 1, nslices)
        m = BatchVariance(nA)
        
        for i in j : batchsize : e
            l = min(i - 1 + batchsize, e)
            tr = i : l
            add!(m, @view(A[:, tr]))
            next!(pbar; step = length(tr))
        end

        m
    end for j in 1 : chunk_size : nslices]

    ret = reduce(add!, fetch.(tasks))

    finish!(pbar)

    return ret
end

function meanvar(samples::AbstractArray{T}, data::AbstractMatrix{UInt8};
                 batchsize = 128,
                 progress = true) where {T}

    ntraces = size(samples, 2)
    nsamples = size(samples, 1)
    ndata = size(data, 1)

    pbar = Progress(ntraces; enabled = progress, showspeed = true)

    chunk_size = cld(ntraces, nthreads())

    tasks = [@spawn begin
        e = min(j + chunk_size - 1, ntraces)

        # Thread-local accumulators
        vars  = [Vector{BatchVariance}(undef, 256) for _ in 1:ndata]
        cache = [Vector{Matrix{T}}(undef, 256) for _ in 1:ndata]
        cacheidx = fill(0, 256, ndata)

        for t in j:e
            s = @view(samples[:, t])
            for i in 1:ndata
                v = data[i, t]
                k = v + 1

                if !isassigned(vars[i], k)
                    vars[i][k] = BatchVariance(nsamples)
                    cache[i][k] = zeros(T, nsamples, batchsize)
                end

                cacheidx[k, i] += 1
                cidx = cacheidx[k, i]
                cache[i][k][:, cidx] = s

                if cidx == batchsize
                    cacheidx[k, i] = 0
                    add!(vars[i][k], cache[i][k])
                end
            end

            next!(pbar)
        end

        # Flush partially filled caches
        for k in CartesianIndices(cacheidx)
            if cacheidx[k] != 0
                jv, i = k.I
                add!(vars[i][jv], @view(cache[i][jv][:, 1:cacheidx[k]]))
            end
        end

        vars
    end for j in 1:chunk_size:ntraces]

    # Collect partial results and combine
    vars_chunks = fetch.(tasks)

    # Reduce across tasks
    result = vars_chunks[1]
    for chunk in Iterators.drop(vars_chunks, 1)
        for i in 1:ndata
            for j in 1:256
                if isassigned(chunk[i], j)
                    if isassigned(result[i], j)
                        add!(result[i][j], chunk[i][j])
                    else
                        result[i][j] = chunk[i][j]
                    end
                end
            end
        end
    end

    # Final compact representation
    avals = [findall(x -> isassigned(r, x), 1:256) .- 1 for r in result]
    avarsvec = [[result[i][x+1] for x in vals] for (i, vals) in enumerate(avals)]

    finish!(pbar)

    return avals, avarsvec
end

function meanvar(samples, data::AbstractMatrix{T};
                    progress = true) where {T}
    ntraces = size(samples, 2)
    nsamples = size(samples, 1)
    ndata   = size(data, 1)

    nthreads = Threads.nthreads()
    chunk_size = cld(ntraces, nthreads)

    pbar = Progress(ntraces; enabled = progress, showspeed = true)

    tasks = [@spawn begin
        # each task has its own independent dictionary array
        local_vars = [Dict{T, BatchVariance}() for _ in 1:ndata]

        jend = min(j + chunk_size - 1, ntraces)
        for t in j:jend
            s = @view(samples[:, t])
            for i in 1:ndata
                v = data[i, t]
                get!(local_vars[i], v, BatchVariance(nsamples))
                add!(local_vars[i][v], s)
            end
            next!(pbar)
        end

        local_vars
    end for j in 1:chunk_size:ntraces]

    # gather and combine results from all tasks
    vars_list = fetch.(tasks)
    combiner(x, y) = (add!(x, y); x)

    result = vars_list[1]
    for k in 2:length(vars_list)
        vars_k = vars_list[k]
        for j in 1:ndata
            mergewith(combiner, result[j], vars_k[j])
        end
    end

    avals = [keys(r) |> collect |> sort for r in result]
    avarsvec = [[result[i][x] for x in vals] for (i, vals) in enumerate(avals)]

    finish!(pbar)

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
