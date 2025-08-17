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
        denom_dfs = ((vx[i]^2 / (nx^2 * (nx - 1))) + 
                     (vy[i]^2 / (ny^2 * (ny - 1))))
        denom_dfs = iszero(denom_dfs) ? eps(Float64) : denom_dfs
        dfs[i] = denom_sq / denom_dfs

        pvals[i] = 2 * (1 - cdf(TDist(dfs[i]), abs(tvals[i])))
    end

    return WelchTResult(tvals, dfs, pvals)
end


struct WelchANOVAResult{T}
    F::Vector{T}
    df1::T
    df2::Vector{T}
    pvalue::Vector{T}
end

export welch_anova
function welch_anova(groups::Vararg{BatchVariance})
    k = length(groups)

    # Extract matrix of means and variances: each column is a group
    means = hcat([getMean(g) for g in groups]...)  # size (d, k)
    vars = hcat([getVariance(g) for g in groups]...)  # size (d, k)
    ns = Float64[nobservations(g) for g in groups]  # vector of length k

    d, _ = size(means)
    weights = ns' .\ vars  # elementwise ns_i / var_ij → size (d, k)
    total_weight = sum(weights, dims=2)  # size (d, 1)
    weighted_mean = sum(weights .* means, dims=2) ./ total_weight  # size (d, 1)

    # Numerator: between-group weighted variance
    A = sum(weights .* (means .- weighted_mean).^2, dims=2) ./ (k - 1)  # size (d, 1)

    # Welch-Satterthwaite denominator correction
    B_term = ((1 .- weights ./ total_weight).^2) ./ (ns .- 1)'  # size (d, k)
    B = sum(B_term, dims=2)  # size (d, 1)
    B_correction = 1 .+ (2 * (k - 2) / (k^2 - 1)) .* B  # size (d, 1)

    F = vec(A ./ B_correction)  # Flatten to Vector{T}
    df1 = Float64(k - 1)
    df2 = vec((k^2 - 1) ./ (3 .* B))  # Welch denominator degrees of freedom

    # p-values per dimension
    p = map((f, df2i) -> 1 - cdf(FDist(df1, df2i), f), F, df2)

    return WelchANOVAResult(F, df1, df2, p)
end
