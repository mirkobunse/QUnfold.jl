using ArgParse, CSV, DataFrames, Printf

select_best(df, metric) = combine(
        groupby(df, :method),
        sdf -> begin
            sdf2 = sdf[.!(ismissing.(sdf[!,metric])),:]
            nrow(sdf2) > 0 ? sdf2[argmin(sdf2[!,metric]),:] : sdf[1,:]
        end
    )

extract_order(x) = Dict(
        "QuaPy" => -1, # will be removed; just computed for reference
        nothing => 1,
        "ovr" => 2,
        "inv" => 3,
        "pinv" => 4,
        "constrained" => 5,
        "softmax" => 6,
        "softmax; Python" => 7,
        "softmax reg." => 8,
        "softmax full reg." => 9,
        "solver=\"dogleg\"" => 101,
        "solver=\"trust-ncg\"" => 102,
        "solver=\"trust-krylov\"" => 103,
        "solver=\"trust-exact\"" => 104,
    )[x[2]]

extract_adjustment(x) = Dict(
        "QuaPy" => "",
        nothing => "un-adjusted (Eq.~\\ref{eq:binary-cc} / Eq.~\\ref{eq:binary-pcc})",
        "ovr" => "one-vs-rest (Eq.~\\ref{eq:ovr})",
        "inv" => "inverse (Eq.~\\ref{eq:inv})",
        "pinv" => "pseudo-inverse (Eq.~\\ref{eq:pinv})",
        "constrained" => "constrained (Eq.~\\ref{eq:constrained})",
        "softmax" => "soft-max with \$[\\vec{l}\\,]_C = 0, \\lambda=0\$",
        "softmax; Python" => "soft-max with \$[\\vec{l}\\,]_C = 0, \\lambda=0\$ (Python)",
        "softmax reg." => "soft-max with \$[\\vec{l}\\,]_C = 0, \\lambda=10^{-6}\$",
        "softmax full reg." => "soft-max with \$[\\vec{l}\\,]_C \\in \\mathbb{R}, \\lambda=10^{-6}\$ (Eq.~\\ref{eq:softmax})",
        "solver=\"dogleg\"" => "dogleg",
        "solver=\"trust-ncg\"" => "trust-ncg",
        "solver=\"trust-krylov\"" => "trust-krylov",
        "solver=\"trust-exact\"" => "trust-exact",
    )[x[2]]

extract_method(x) = Dict(
        "EMQ" => "",
        "ACC" => "ACC / CC",
        "CC" => "ACC / CC",
        "PACC" => "PACC / PCC",
        "PCC" => "PACC / PCC"
    )[x[1]]

format_scores(x) =
    [ (x_i==minimum(x) ? "\$\\mathbf{" : "\${") * @sprintf("%.4f", x_i) * "}\$" for x_i ∈ x ]

function table_for_metric(df_val, df_tst, metric)
    df_val = select_best(df_val, metric)
    df = leftjoin(df_val[!,[:method, :C]], df_tst, on=[:method, :C]) # mirror the selection
    @info "Selected the best methods for the metric $(metric)" df_val df
    matches = match.(r"(\w+)(?: \((.+)\))?", df[!,:method]) # separate method from adjustment
    df[!,:adjustment] = extract_adjustment.(matches)
    df[!,:order] = extract_order.(matches)
    df[!,:method] = extract_method.(matches)
    df = sort(df[df[!,:order] .> 0,:], :order) # remove reference methods
    df = unstack(df, :adjustment, :method, metric)
    df[!,"ACC / CC"] = format_scores(df[!,"ACC / CC"])
    df[!,"PACC / PCC"] = format_scores(df[!,"PACC / PCC"])
    return df
end

export_table(output_path, df) = open(output_path, "w") do io
    println(io, "\\begin{tabular}{l$(repeat("c", size(df, 2)-1))}")
    println(io, "  \\toprule")
    println(io, "    ", join(names(df), " & "), " \\\\") # header
    println(io, "  \\midrule")
    for r in eachrow(df)
        println(io, "    ", join(r, " & "), " \\\\")
    end
    println(io, "  \\bottomrule")
    println(io, "\\end{tabular}")
end

function main(; validation_path::String="", testing_path::String="", output_path::String="")
    @info "Reading from $(validation_path) and $(testing_path)"
    df_val = CSV.read(validation_path, DataFrame)
    df_tst = CSV.read(testing_path, DataFrame)
    @info "Configurations with failures" df_val[(df_val[!,:n_pdf_failures].>0) .| (df_val[!,:n_failures].>0),:]
    df = innerjoin(
        table_for_metric(df_val, df_tst, :ae),
        table_for_metric(df_val, df_tst, :rae),
        on = :adjustment,
        makeunique = true,
        renamecols = Pair("; AE", "; RAE")
    )
    df = df[:, [ # re-order the columns
        :adjustment,
        Symbol("ACC / CC; AE"),
        Symbol("ACC / CC; RAE"),
        Symbol("PACC / PCC; AE"),
        Symbol("PACC / PCC; RAE"),
    ]]
    export_table(output_path, df)
    @info "LaTeX table exported to $(output_path)"
end

# command line interface
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "validation_path"
            help = "the output path of the validation results"
            required = true
        "testing_path"
            help = "the output path of the testing results"
            required = true
        "output_path"
            help = "the output path for the LaTeX table"
            required = true
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; parse_commandline()...)
end
