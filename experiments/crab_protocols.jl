if ".." ∉ LOAD_PATH push!(LOAD_PATH, "..") end # add QUnfold to the LOAD_PATH
using ArgParse, DataFrames, Random, QUnfoldExperiments

function main(;
        curvature_path :: String = "results/crab_protocols_Tp.tex",
        shift_path :: String = "results/crab_protocols_nmd.tex",
        log10 :: Bool = true,
        )
    Random.seed!(876) # make this experiment reproducible
    df_curvature = DataFrame(
        Symbol("N") => Int[],
        Symbol("protocol") => String[],
        Symbol("5\\textsuperscript{th}") => String[],
        Symbol("25\\textsuperscript{th}") => String[],
        Symbol("50\\textsuperscript{th}") => String[],
        Symbol("75\\textsuperscript{th}") => String[],
        Symbol("95\\textsuperscript{th}") => String[],
        Symbol("average") => String[],
    )
    df_shift = copy(df_curvature) # same columns
    m = 10000 # number of samples generated by each protocol
    for N in [1000, 10000]
        p_trn = if log10
            QUnfoldExperiments.to_log10_spectrum_density(N, QUnfoldExperiments.P_TRN[])
        else
            QUnfoldExperiments.P_TRN[]
        end
        for (protocol, samples) in [
                "APP" => QUnfoldExperiments.sample_app(N, m),
                "APP-OQ (20\\%)" => QUnfoldExperiments.sample_app_oq(N, m, .2),
                "APP-OQ (5\\%)" => QUnfoldExperiments.sample_app_oq(N, m, .05),
                "APP-OQ (1\\%)" => QUnfoldExperiments.sample_app_oq(N, m, .01),
                "NPP (Crab)" => QUnfoldExperiments.sample_npp_crab(N, m),
                "NPP (simulation)" => QUnfoldExperiments.sample_npp_simulation(N, m),
                "Poisson" => QUnfoldExperiments.sample_poisson(N, m),
            ]
            spectra = if log10
                [ QUnfoldExperiments.to_log10_spectrum_density(N, p) for p in samples ]
            else
                samples
            end
            push!(df_curvature, vcat(
                N,
                protocol,
                QUnfoldExperiments.format_statistics([
                    QUnfoldExperiments.curvature(p)
                    for p in spectra
                ])
            ))
            push!(df_shift, vcat(
                N,
                protocol,
                QUnfoldExperiments.format_statistics([
                    QUnfoldExperiments.nmd(p, p_trn)
                    for p in spectra
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
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table s begin
        "--no-log10"
            help = "whether to omit the log10 transformation and acceptance correction"
            action = :store_true
        "curvature_path"
            help = "the output path for the curvature table"
            required = true
        "shift_path"
            help = "the output path for the prior probability shift table"
            required = true
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    clargs = parse_commandline()
    clargs[:log10] = !pop!(clargs, :no_log10, false)
    main(; clargs...)
end
