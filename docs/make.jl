using Documenter, QUnfold

makedocs(;
    sitename = "QUnfold.jl",
    pages = [
        "Home" => "index.md",
        "API reference" => "api-reference.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/mirkobunse/QUnfold.jl.git",
    devbranch = "main"
)
