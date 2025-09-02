using Statistics, Distributions

struct WelchTResult{T}
    t::Vector{T}
    df::Vector{T}
    pvalue::Vector{T}
end

export welch_t
function welch_t(x::BatchVariance, y::BatchVariance)
    mx = getMean(x)
    my = getMean(y)
    vx = getVariance(x)
    vy = getVariance(y)

    length(mx) == length(my) || error("unequal length of means: $(length(mx)) != $(length(my))")
    length(vx) == length(vy) || error("unequal length of variances: $(length(vx)) != $(length(vy))")

    nx = nobservations(x)
    ny = nobservations(y)

    l = 2
    nx > l || error("need at least $l observations for group 0 (have $nx)")
    ny > l || error("need at least $l observations for group 1 (have $ny)")

    d = length(mx)
    tvals = Vector{eltype(mx)}(undef, d)
    dfs = Vector{eltype(mx)}(undef, d)
    pvals = Vector{eltype(mx)}(undef, d)

    for i in 1:d
        vxn = vx[i] / nx
        vyn = vy[i] / ny
        denom = vxn + vyn
        denom = iszero(denom) ? eps(Float64) : denom

        tvals[i] = (mx[i] - my[i]) / sqrt(denom)

        denom_sq = denom^2
        denom_dfs = ((vx[i]^2 / (Float64(nx)^2 * (nx - 1))) + 
                     (vy[i]^2 / (Float64(ny)^2 * (ny - 1))))
        denom_dfs = iszero(denom_dfs) ? eps(Float64) : denom_dfs
        dfs[i] = denom_sq / denom_dfs

        pvals[i] = 2 * (1 - cdf(TDist(dfs[i]), abs(tvals[i])))
    end

    return WelchTResult(tvals, dfs, pvals)
end

export welcht_one_vs_rest
function welcht_one_vs_rest(vars::AbstractVector{T}) where {T <: BatchVariance}
    nsamples = length(first(vars).varx)
    nvars = length(vars)

    others = [BatchVariance(nsamples) for i in 1 : nvars]

    for i in 1 : nvars
        for j in 1 : nvars
            if i == j
                continue
            end

            add!(others[i], vars[j])
        end
    end

    tmatrix = zeros(nsamples, nvars)

    for i in 1 : nvars
        wt = welch_t(others[i], vars[i])
        tmatrix[:, i] .= wt.t
    end

    return tmatrix
end

export welcht_pairwise
function welcht_pairwise(vars::AbstractVector{T}) where {T <: BatchVariance}
    nsamples = length(first(vars).varx)
    nvals = length(vars)
    tmatrix = zeros(nsamples, nvals, nvals)

    for i in 1 : nvals - 1
        var1 = vars[vals[i]]
        for j in i + 1 : nvals
            var2 = vars[vals[j]]
            wt = welch_t(var1, var2)
            tmatrix[:, i, j] = wt.t
        end
    end

    return tmatrix
end
