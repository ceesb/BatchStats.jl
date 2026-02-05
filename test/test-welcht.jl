using Test
using HypothesisTests
using BatchStats


function testwelcht()
    # Create two random samples with different means
    X = randn(3, 100) .+ 1.0   # 3 variables, 100 samples
    Y = randn(3, 80) .+ 0.0

    # Wrap in test-compatible BatchVariance types
    batchsize = 16
    x = BatchVariance(size(X, 1))
    y = BatchVariance(size(Y, 1))

    for i in 1 : batchsize : size(X, 2)
        l = min(size(X, 2), i - 1 + batchsize)
        add!(x, @view(X[:, i : l]))
    end

    for i in 1 : batchsize : size(Y, 2)
        l = min(size(Y, 2), i - 1 + batchsize)
        add!(y, @view(Y[:, i : l]))
    end

    # Call your implementation
    result = welcht(x, y)

    # Compare each variable independently with HypothesisTests
    for i in 1:size(X,1)
        ttest = UnequalVarianceTTest(X[i, :], Y[i, :])
        @test isapprox(result.t[i], ttest.t)
        @test isapprox(result.df[i], ttest.df)
        @test isapprox(result.pvalue[i], pvalue(ttest), atol=1e-8)
    end
end
