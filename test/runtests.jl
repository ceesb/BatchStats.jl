using Test

@testset "pearson stats" begin
include("test-pearson.jl")
testpearson()
testpearson_nulladd()
end

@testset "variance stats" begin
include("test-variance.jl")
testvariance()
testvariance_nulladd()
end

@testset "welch_t stats" begin
include("test-welcht.jl")
testwelcht()
end

@testset "test wrappers" begin
include("test-wrappers.jl")
testvar()
testcor()
testmeanvar()

end