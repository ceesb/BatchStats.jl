[![](https://img.shields.io/badge/docs-green.svg)](https://ceesb.github.io/BatchStats.jl/)

# BatchStats

A few first-order statistics in Julia supporting three update modes: incremental (single observations), batch (multiple observations at once), and aggregate (combining statistics). Constant memory complexity independent of the number of observations.

Supports:
- mean / variance
- covariance / pearson's correlation
- welch's t-statistic

These implementations are constant memory, i.e. the internal state size is independent of the number of observations fed into it. There are other Julia packages that do (way more) incremental statistics, for example `OnlineStats`. The difference with `OnlineStats` that it does not do batch updates. Updates of batches of observations are faster than single observation updates.

# Examples

First, a Pearson correlation example.

```julia
using BatchStats

nx = 13
ny = 15
batchsize = 128

ic = BatchCorrelation(nx, ny, batchsize)

# incremental update
add!(ic, rand(nx), rand(ny))

# batch update
add!(ic, rand(nx, batchsize), rand(ny, batchsize))

# you can add less than batchsize but not more, in one call
add!(ic, rand(nx, batchsize - 10), rand(ny, batchsize - 10))

# get the correlation statistic
Cr = getCorrelation(ic)

# or the covariance matrix
Cv = getCovariance(ic)

# or the means and variances
Mx = getMeanX(ic)
My = getMeanY(ic)
Vx = getVarianceX(ic)
Vy = getVarianceY(ic)
```

If you just want a replacement for the `Statistics` correlation `cor` (or variance `var`), you can do the following. Note that `cor` and `var` are not exported so to not conflict the `Statistics` versions.

```julia
import BatchStats

nx = 100
ny = 75
nobservations = 1000
X = rand(nx,nobservations)
Y = rand(ny,nobservations)
result = BatchStats.cor(X, Y; dims = 2)
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

On a Mac Air M3 from 2023 `Stastistics.cor` starts swapping, but the `BatchStats.cor` still makes it.
```
julia> versioninfo()
Julia Version 1.11.6
Commit 9615af0f269 (2025-07-09 12:58 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: macOS (arm64-apple-darwin24.0.0)
  CPU: 8 × Apple M3
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, apple-m2)
Threads: 1 default, 0 interactive, 1 GC (on 4 virtual cores)

julia> @time BatchStats.cor(X, Y; dims = 2, batchsize = 2048); nothing
100.757170 seconds (53 allocations: 800.315 MiB, 0.06% gc time)
```
