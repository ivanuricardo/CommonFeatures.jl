using CommonFeatures
using Documenter

DocMeta.setdocmeta!(CommonFeatures, :DocTestSetup, :(using TensorEconometrics); recursive=true)

makedocs(;
    modules=[CommonFeatures],
    authors="Ivan Ricardo <iu.ricardo@maastrichtuniversity.nl> and contributors",
    repo="https://github.com/ivanuricardo/CommonFeatures.jl/blob/{commit}{path}#{line}",
    sitename="CommonFeatures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ivanuricardo.github.io/CommonFeatures.jl",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo="github.com/ivanuricardo/CommonFeatures.jl",
    devbranch="main"
)
