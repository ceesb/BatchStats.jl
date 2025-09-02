using LinearAlgebra, Statistics

struct BatchCorrelation{T}
    n::Base.RefValue{Int}
    meanx::Vector{T}
    meany::Vector{T}
    varx::Vector{T}
    vary::Vector{T}
    mx::Vector{T}
    my::Vector{T}
    Δx::Vector{T}
    Δy::Vector{T}
    cov::Matrix{T}
    nbatch::Int
    Xc::Matrix{T}
    Yc::Matrix{T}
end

export reset!
function reset!(this::BatchCorrelation)
    this.n[] = 0
    fill!(this.meanx, 0)
    fill!(this.meany, 0)
    fill!(this.varx, 0)
    fill!(this.vary, 0)
    fill!(this.mx, 0)
    fill!(this.my, 0)
    fill!(this.Δx, 0)
    fill!(this.Δy, 0)
    fill!(this.cov, 0)
    fill!(this.Xc, 0)
    fill!(this.Yc, 0)
end

export BatchCorrelation
function BatchCorrelation(nx::Integer, ny::Integer, nbatch::Integer=16, ::Type{T}=Float64) where T
    BatchCorrelation{T}(
        Ref(0),
        zeros(T, nx),
        zeros(T, ny),
        zeros(T, nx),
        zeros(T, ny),
        zeros(T, nx),
        zeros(T, ny),
        zeros(T, nx),
        zeros(T, ny),
        zeros(T, nx, ny),
        nbatch,
        zeros(T, nx, nbatch),
        zeros(T, ny, nbatch),
    )
end

export add!
function add!(ic::BatchCorrelation{T}, x::AbstractVector, y::AbstractVector) where T
    size(x, 1) == size(ic.meanx, 1) || error("Wrong length x $(size(x, 1)) != $(size(ic.meanx, 1))") 
    size(y, 1) == size(ic.meany, 1) || error("Wrong length y $(size(y, 1)) != $(size(ic.meany, 1))") 

    ic.n[] += 1
    α = (ic.n[] - 1) / ic.n[]
    β = 1 / ic.n[]
    meanx, meany = ic.meanx, ic.meany
    dx, dy = ic.Δx, ic.Δy

    @. dx = x - meanx
    @. dy = y - meany
    @. meanx += β * dx
    @. meany += β * dy

    mul!(ic.cov, dx, dy', α, 1)  # cov += α * dx * dy'

    @. ic.varx += α * dx * dx
    @. ic.vary += α * dy * dy  

    return ic
end

function add!(ic::BatchCorrelation{T}, X::AbstractMatrix, Y::AbstractMatrix) where T
    size(X, 1) == size(ic.meanx, 1) || error("Wrong length x $(size(X, 1)) != $(size(ic.meanx, 1))") 
    size(Y, 1) == size(ic.meany, 1) || error("Wrong length y $(size(Y, 1)) != $(size(ic.meany, 1))") 
    size(X, 2) == size(Y, 2) || error("Different number of observations between x and y: $(size(X, 2)) != $(size(Y, 2))")
    nbatch = size(X, 2)
    nbatch <= ic.nbatch || error("Too many number of observations in batch: $(nbatch) > $(ic.nbatch)")

    if nbatch == 0
        return
    end

    n1 = ic.n[]
    n2 = nbatch
    n = n1 + n2
    ic.n[] = n

    β = (-n2 / n) ^ 2 * n1 + (n1 / n) ^ 2 * n2

    meanx, meany = ic.meanx, ic.meany
    mx, my = ic.mx, ic.my
    Δx, Δy = ic.Δx, ic.Δy
    Xc, Yc = @views ic.Xc[:, 1:nbatch], @views ic.Yc[:, 1:nbatch]
    Xv, Yv = @views X[:, 1:nbatch], @views Y[:, 1:nbatch]

    mean!(mx, Xv)
    mean!(my, Yv)

    @. Δx = mx - meanx
    @. Δy = my - meany

    @. meanx += (n2 / n) * Δx
    @. meany += (n2 / n) * Δy

    @. Xc = Xv - mx
    @. Yc = Yv - my

    mul!(ic.cov, Xc, Yc', 1, 1)  # cov += (Xc * Yc')
    mul!(ic.cov, Δx, Δy', β, 1)  # cov += β * Δx * Δy'

    # @. Xc .*= Xc
    sum!(abs2, mx, Xc)
    @. ic.varx += mx
    @. ic.varx += β * Δx * Δx

    # @. Yc .*= Yc
    sum!(abs2, my, Yc)
    @. ic.vary += my
    @. ic.vary += β * Δy * Δy

    return ic
end


function add!(ic::BatchCorrelation{T}, other::BatchCorrelation{T}) where T
    n1, n2 = ic.n[], other.n[]

    if n2 == 0
        return
    end
    
    n = n1 + n2
    α = n2 / n
    β = (-n2 / n) ^ 2 * n1 + (n1 / n) ^2 * n2

    Δx = ic.Δx
    Δy = ic.Δy
    @. Δx = other.meanx - ic.meanx
    @. Δy = other.meany - ic.meany

    @. ic.meanx += Δx * α
    @. ic.meany += Δy * α

    ic.n[] = n

    mul!(ic.cov, Δx, Δy', β, 1)   # cov += β * Δx * Δy'
    @. ic.cov += other.cov

    @. ic.varx += β * Δx * Δx
    @. ic.varx += other.varx

    @. ic.vary += β * Δy * Δy
    @. ic.vary += other.vary

    return ic
end

export getMeanX
function getMeanX(ic::BatchCorrelation)
    return ic.meanx
end

export getVarianceX
function getVarianceX(ic::BatchCorrelation)
    v = 1 / (ic.n[] - 1) .* ic.varx
    return v
end

export getMeanY
function getMeanY(ic::BatchCorrelation)
    return ic.meany
end

export getVarianceY
function getVarianceY(ic::BatchCorrelation)
    v = 1 / (ic.n[] - 1) .* ic.vary
    return v
end

export getCovariance
function getCovariance(ic::BatchCorrelation{T}; corrected = true) where {T}
    if corrected
        return ic.cov .* 1/(ic.n[] - 1)
    else
        return ic.cov .* 1/(ic.n[])
    end
end

export getCorrelation
function getCorrelation(ic::BatchCorrelation{T}) where T
    stdx = sqrt.(getVarianceX(ic))
    stdy = sqrt.(getVarianceY(ic))
    C = getCovariance(ic)

    stdx[iszero.(stdx)] .= eps(Float64)
    stdy[iszero.(stdy)] .= eps(Float64)
    
    rdiv!(C, Diagonal(stdy))
    ldiv!(Diagonal(stdx), C)

    return C
end

export nobservations
nobservations(ic::BatchCorrelation) = ic.n[]

