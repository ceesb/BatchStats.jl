struct LazyDummyArray{T, N} <: AbstractArray{T, N}
    dims::NTuple{N, Int}
    seed::UInt
end

# Convenience constructor
function LazyDummyArray(dims::NTuple{N, Int}, seed=0x12345678) where N
    LazyDummyArray{Float64, N}(dims, UInt(seed))
end

# Array interface
Base.size(A::LazyDummyArray) = A.dims
Base.axes(A::LazyDummyArray) = ntuple(i -> Base.OneTo(A.dims[i]), length(A.dims))
Base.length(A::LazyDummyArray) = prod(A.dims)
Base.ndims(A::LazyDummyArray) = length(A.dims)
Base.eltype(::LazyDummyArray{T}) where T = T
Base.IndexStyle(::Type{<:LazyDummyArray}) = IndexLinear()

# Fast deterministic hash
@inline function simple_hash(seed::UInt, i::Int)
    h = seed ⊻ UInt(i)
    h *= 0x9e3779b97f4a7c15
    h = h ⊻ (h >> 32)
    return h
end

# Convert to Float64 in [0, 1)
@inline function hash_to_float64(h::UInt)
    mant = h >> (64 - 52)
    return reinterpret(Float64, 0x3FF0000000000000 | mant) - 1.0
end

# Linear indexing only (IndexLinear style)
@inline function Base.getindex(A::LazyDummyArray{Float64, N}, i::Int) where N
    h = simple_hash(A.seed, i)
    return hash_to_float64(h)
end
