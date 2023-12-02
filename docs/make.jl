using TensorEconometrics
using Documenter

DocMeta.setdocmeta!(TensorEconometrics, :DocTestSetup, :(using TensorEconometrics); recursive=true)

makedocs(;
    modules=[TensorEconometrics],
    authors="Ivan Ricardo <iu.ricardo@maastrichtuniversity.nl> and contributors",
    repo="https://github.com/ivanuricardo/TensorEconometrics.jl/blob/{commit}{path}#{line}",
    sitename="TensorEconometrics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ivanuricardo.github.io/TensorEconometrics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ivanuricardo/TensorEconometrics.jl",
    devbranch="main",
)
