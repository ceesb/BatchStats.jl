# BatchStats

A few first-order statistics in Julia supporting incremental (1 sample), batch (n samples), aggregate (a whole other state) updates.

Supports:
- mean / variance
- covariance / pearson's correlation
- welch's t-statistic

These implementations are constant memory, i.e. the state size is independent of the number of observations you're feeding into it. There are other Julia packages that do (way more) incremental statistics, for example `OnlineStats`. The difference with `OnlineStats` that it does not do batch updates. Batch updates are faster than single updates.

# Examples

XXX FIXME

```julia
import Statistics
import BatchStats

nx = 100
ny = 75
nobservations = 1000
X = rand(nx,nobservations)
Y = rand(ny,nobservations)
result = BatchStats.cor(X, Y; dims = 2)

@assert Statistics.cor(X, Y; dims = 2) ≈ BatchStats.getCorrelation(result)
@time Statistics.cor(X, Y; dims = 2)
@time BatchStats.cor(X, Y; dims = 2)
```

# Performance

Let's compare the `cor` correlation function performance againsts `Statistics`. Time-wise we can get close, memory wise we will win. 

We'll use a large dummy array: the contents are deterministic but not stored in memory.

```julia
import Statistics
import BatchStats
include(pkgdir(BatchStats) * "/test/lazydummyarray.jl")

nx = 10_000
ny = 7_000
nobservations = 100_000
X = LazyDummyArray((nx,nobservations), 0x42)
Y = LazyDummyArray((ny,nobservations), 0x42)

# run both for precompilation and compare outputs
result = BatchStats.cor(X, Y; dims = 2, batchsize = 2048)
@assert Statistics.cor(X, Y; dims = 2) ≈ BatchStats.getCorrelation(result)

@time Statistics.cor(X, Y; dims = 2); nothing
@time BatchStats.cor(X, Y; dims = 2, batchsize = 2048); nothing
```

On an Intel i9-14900K I see this:
```
julia> versioninfo()
Julia Version 1.11.6
Commit 9615af0f269 (2025-07-09 12:58 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 32 × Intel(R) Core(TM) i9-14900K
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, alderlake)
Threads: 1 default, 0 interactive, 1 GC (on 32 virtual cores)

julia> @time Statistics.cor(X, Y; dims = 2); nothing
 36.750268 seconds (33 allocations: 13.188 GiB, 0.07% gc time)

julia> @time BatchStats.cor(X, Y; dims = 2, batchsize = 2048); nothing
 36.121666 seconds (53 allocations: 800.212 MiB, 0.70% gc time)
```

# Lower level functions

XXX