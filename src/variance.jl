using LinearAlgebra, Statistics

struct BatchVariance{T}
    n::Base.RefValue{Int}
    meanx::Vector{T}
    varx::Vector{T}
    mx::Vector{T}
    Δx::Vector{T}
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
function add!(ic::BatchVariance{T}, x::AbstractVector) where T
    size(x, 1) == size(ic.meanx, 1) || error("Wrong length x $(size(x, 1)) != $(size(ic.meanx, 1))") 

    ic.n[] += 1
    α = (ic.n[] - 1) / ic.n[]
    β = 1 / ic.n[]
    meanx = ic.meanx
    dx = ic.Δx

    @. dx = x - meanx
    @. meanx += β * dx
    @. ic.varx += α * dx * dx

    return ic
end

function add!(ic::BatchVariance{T}, X::AbstractMatrix) where T
    size(X, 1) == size(ic.meanx, 1) || error("Wrong length x $(size(X, 1)) != $(size(ic.meanx, 1))") 
    nbatch = size(X, 2)

    if nbatch == 0
        return
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


function add!(ic::BatchVariance{T}, other::BatchVariance{T}) where T
    n1, n2 = ic.n[], other.n[]
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
function getMean(ic::BatchVariance)
    return ic.meanx
end

export getVariance
function getVariance(ic::BatchVariance)
    v = 1 / (ic.n[] - 1) .* ic.varx
    return v
end

export nobservations
nobservations(ic::BatchVariance) = ic.n[]

