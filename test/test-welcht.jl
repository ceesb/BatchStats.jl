using Test
using HypothesisTests
using BatchStats


function testwelcht()
    @testset "welch_t stats" begin
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
        result = welch_t(x, y)

        # Compare each variable independently with HypothesisTests
        for i in 1:size(X,1)
            ttest = UnequalVarianceTTest(X[i, :], Y[i, :])
            @test isapprox(result.t[i], ttest.t)
            @test isapprox(result.df[i], ttest.df)
            @test isapprox(result.pvalue[i], pvalue(ttest), atol=1e-8)
        end
    end
end

function testwelchanova()
    @testset "welch_anova vs HypothesisTests (approximate)" begin
        # Three groups of multivariate data (3 features)
        G1 = randn(3, 100) .+ 0.0
        G2 = randn(3, 80) .+ 1.0
        G3 = randn(3, 60) .+ 2.0

        G1 = G1 ./ std(G1)
        G2 = G2 ./ std(G2)
        G3 = G3 ./ std(G3)

        batchsize = 16
        groups = BatchVariance[
            let x = BatchVariance(size(G1, 1))
                for i in 1:batchsize:size(G1, 2)
                    l = min(size(G1, 2), i - 1 + batchsize)
                    add!(x, @view(G1[:, i:l]))
                end
                x
            end,
            let x = BatchVariance(size(G2, 1))
                for i in 1:batchsize:size(G2, 2)
                    l = min(size(G2, 2), i - 1 + batchsize)
                    add!(x, @view(G2[:, i:l]))
                end
                x
            end,
            let x = BatchVariance(size(G3, 1))
                for i in 1:batchsize:size(G3, 2)
                    l = min(size(G3, 2), i - 1 + batchsize)
                    add!(x, @view(G3[:, i:l]))
                end
                x
            end
        ]

        result = welch_anova(groups...)

        # Basic sanity checks
        @test length(result.F) == 3
        @test result.df1 == 2
        @test all(result.df2 .> 0)
        @test all(result.pvalue .>= 0)
        @test all(result.pvalue .<= 1)

        # Rough comparison with classical OneWayANOVA when variances are similar
        for i in 1:3
            aov = OneWayANOVATest([G1[i, :], G2[i, :], G3[i, :]]...)
            # Welch F should generally be close in large samples if variances are not too different
            @test isapprox(result.F[i], HypothesisTests.teststatistic(aov), atol=0.5)  # allow fuzziness
            @test isapprox(result.pvalue[i], pvalue(aov), atol=0.1)
        end
    end
end


