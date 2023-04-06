if ".." ∉ LOAD_PATH push!(LOAD_PATH, "..") end # add QUnfold to the LOAD_PATH
using ArgParse, DataFrames, Distributions, LinearAlgebra, Random, QUnfoldExperiments

P_TRN = [ 0.09345, 0.07065, 0.09385, 0.16, 0.58205 ] # from github.com:mirkobunse/ecml22

function curvature(p::AbstractVector{T}) where T<:Number
    t = LinearAlgebra.diagm( # Tikhonov matrix
        -1 => fill(-1, length(P_TRN)-1),
        0 => fill(2, length(P_TRN)),
        1 => fill(-1, length(P_TRN)-1)
    )[2:(length(P_TRN)-1), :]
    return (t * p)' * (t * p) / 2 # 1/2 (Tp)^2
end

sample_npp(N, m) = [ sample_npp(N) for _ in 1:m ]
sample_npp(N) = QUnfoldExperiments.round_Np(N, P_TRN) ./ N

sample_app(N, m) = [ sample_app(N) for _ in 1:m ]
sample_app(N) = QUnfoldExperiments.round_Np(N, rand(Dirichlet(ones(length(P_TRN))))) ./ N

function sample_app_oq(N, m=10000, keep=.2)
    app = sample_app(N, ceil(Int, m/keep))
    c = [ curvature(x) for x in app ]
    i = sortperm(c)[1:m]
    return app[i]
end

function main(;
        curvature_path :: String = "results/amazon_protocols_Tp.tex",
        shift_path :: String = "results/amazon_protocols_nmd.tex",
        )
    Random.seed!(876) # make this experiment reproducible
    df_curvature = DataFrame(
        Symbol("N") => String[ "" ],
        Symbol("protocol") => String[ "products" ],
        Symbol("5\\textsuperscript{th}") => String[ "0.01624" ],
        Symbol("25\\textsuperscript{th}") => String[ "0.04745" ],
        Symbol("50\\textsuperscript{th}") => String[ "0.09304" ],
        Symbol("75\\textsuperscript{th}") => String[ "0.16480" ],
        Symbol("95\\textsuperscript{th}") => String[ "0.30730" ],
        Symbol("average") => String[ "0.11837" ],
    )
    df_shift = copy(df_curvature) # same columns
    m = 10000 # number of samples generated by each protocol
    for N in [1000, 10000]
        for (protocol, samples) in [
                "APP" => sample_app(N, m),
                "APP-OQ (80\\%)" => sample_app_oq(N, m, .8),
                "APP-OQ (66\\%)" => sample_app_oq(N, m, .66),
                "APP-OQ (50\\%)" => sample_app_oq(N, m, .5),
                "APP-OQ (33\\%)" => sample_app_oq(N, m, .33),
                "APP-OQ (20\\%)" => sample_app_oq(N, m, .2),
                "APP-OQ (5\\%)" => sample_app_oq(N, m, .05),
                "APP-OQ (1\\%)" => sample_app_oq(N, m, .01),
                "NPP" => sample_npp(N, m),
            ]
            push!(df_curvature, vcat(
                "$N",
                protocol,
                QUnfoldExperiments.format_statistics([
                    curvature(p)
                    for p in samples
                ])
            ))
            push!(df_shift, vcat(
                "$N",
                protocol,
                QUnfoldExperiments.format_statistics([
                    QUnfoldExperiments.nmd(p, P_TRN)
                    for p in samples
                ])
            ))
        end
    end
    QUnfoldExperiments.export_table(curvature_path, df_curvature)
    QUnfoldExperiments.export_table(shift_path, df_shift)
    @info "LaTeX tables exported to $(curvature_path) and $(shift_path)"
end

# command line interface
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--curvature_path"
            help = "the output path for the curvature table"
            default = "results/amazon_protocols_Tp.tex"
        "--shift_path"
            help = "the output path for the prior probability shift table"
            default = "results/amazon_protocols_nmd.tex"
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; parse_commandline()...)
end
