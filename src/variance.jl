using LinearAlgebra, Statistics

struct BatchVariance{T}
    n::Base.RefValue{Int}
    meanx::Vector{T}
    varx::Vector{T}
    mx::Vector{T}
    Δx::Vector{T}
end

function Base.show(io::IO, x::BatchVariance)
    nobs = nobservations(x)
    nx = length(x.mx)
    print(io, "BatchVariance, $(nx) samples, $(nobs) observations")
end

export reset!
function reset!(this::BatchVariance)
    this.n[] = 0
    fill!(this.meanx, 0)
    fill!(this.varx, 0)
    fill!(this.mx, 0)
    fill!(this.Δx, 0)
end

export BatchVariance
"""
`BatchVariance(nx::Integer)`

Initializes the BatchVariance state for `nx` samples.
"""
function BatchVariance(nx::Integer, ::Type{T}=Float64) where T
    BatchVariance{T}(
        Ref(0),
        zeros(T, nx),
        zeros(T, nx),
        zeros(T, nx),
        zeros(T, nx),
    )
end

export add!
"""
`add!(ic::BatchVariance, x::AbstractVector)`

Updates the variance statistics with a single observation of `x`.
"""
function add!(ic::BatchVariance{T}, x::AbstractVector) where T
    size(x, 1) == size(ic.meanx, 1) || error("Wrong length x $(size(x, 1)) != $(size(ic.meanx, 1))") 

    ic.n[] += 1
    α = (ic.n[] - 1) / ic.n[]
    β = 1 / ic.n[]
    meanx = ic.meanx
    dx = ic.Δx

    @. dx = x - meanx
    @. meanx += β * dx
    @. ic.varx += α * dx ^ 2

    return ic
end

export add!
"""
`add!(ic::BatchVariance, X::AbstractMatrix)`

Updates the variance statistics with a batch of observation of `X`. Every column is an observation.
"""
function add!(ic::BatchVariance{T}, X::AbstractMatrix) where T
    size(X, 1) == size(ic.meanx, 1) || error("Wrong length x $(size(X, 1)) != $(size(ic.meanx, 1))") 
    nbatch = size(X, 2)

    if nbatch == 0
        return ic
    end

    n1 = ic.n[]
    n2 = nbatch
    n = n1 + n2
    ic.n[] = n

    α = n2 / n
    β = (-n2 / n) ^ 2 * n1 + (n1 / n) ^ 2 * n2

    meanx = ic.meanx
    mx = ic.mx
    Xv = @views X[:, 1:nbatch]

    mean!(mx, Xv)

    @inbounds for j in axes(Xv, 2)
        for i in axes(Xv, 1)
            ic.varx[i] += (X[i,j] - mx[i]) ^ 2
        end
    end

    @inbounds for i in axes(Xv, 1)
        Δm = (mx[i] - meanx[i])
        ic.varx[i] += β * Δm ^ 2
        ic.meanx[i] += α * Δm
    end

    return ic
end

export add!
"""
`add!(ic::BatchVariance, other::BatchVariance)`

Updates the variance statistics in `ic` with the statistics in `other`.
"""
function add!(ic::BatchVariance{T}, other::BatchVariance{T}) where T
    n1, n2 = ic.n[], other.n[]

    if n2 == 0
        return ic
    end
    
    n = n1 + n2
    α = n2 / n
    β = (-n2 / n) ^ 2 * n1 + (n1 / n) ^2 * n2

    Δx = ic.Δx
    @. Δx = other.meanx - ic.meanx

    @. ic.meanx += Δx * α

    ic.n[] = n

    @. ic.varx += β * Δx * Δx
    @. ic.varx += other.varx

    return ic
end

export getMean
"""
Returns the mean
"""
function getMean(ic::BatchVariance)
    return ic.meanx
end

export getVariance
"""
Returns the variance
"""
function getVariance(ic::BatchVariance)
    v = 1 / (ic.n[] - 1) .* ic.varx
    return v
end

export nobservations
nobservations(ic::BatchVariance) = ic.n[]

