# docs/make.jl
using Documenter, TRSFiles

makedocs(
    sitename = "BatchStats.jl",
    modules = [BatchStats],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API"  => "api.md",
    ]
)
