# BatchStats

A few first-order statistics in Julia supporting batch, incremental and aggregate updates.

Supports:
- mean / variance
- covariance / pearson's correlation
- welch's t-statistic

These implementations are constant memory, i.e. the state size is independent of the number of observations you're feeding into it. There are other Julia packages that do incremental statistics, for example `OnlineStats`. The difference with `OnlineStats` is that we currently do way less statistical methods, but more importantly for my usecase is that `OnlineStats` does not do batch updates, and therefore is much slower.

# Examples

Let's compare the `cor` correlation function againsts `Statistics`. Note that `Statistics` is effectly operating on a 'single batch', so we cannot beats its time when single threaded, but we will win in the number of allocations. 

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

Let's make some huge virtual dummy array that does not fit in RAM. We can no longer run the `Statistics` function because it will run out of memory.

```julia
import Statistics
import BatchStats
include(pkgdir(BatchStats) * "/test/lazydummyarray.jl")

nx = 10_000
ny = 7_000
nobservations = 1_000_000
X = LazyDummyArray((nx,nobservations), 0x42)
Y = LazyDummyArray((ny,nobservations), 0x42)

result = BatchStats.cor(X, Y; dims = 2)

@assert Statistics.cor(X, Y; dims = 2) ≈ BatchStats.getCorrelation(result)
@time Statistics.cor(X, Y; dims = 2)
@time BatchStats.cor(X, Y; dims = 2)
```

# Lower level functions

XXX