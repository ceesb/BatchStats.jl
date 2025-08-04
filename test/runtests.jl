include("test-pearson.jl")
testpearson()

include("test-variance.jl")
testvariance()

include("test-welcht.jl")
testwelcht()
# FIXME: broken
# testwelchanova()

include("test-wrappers.jl")
testvar()
testcor()