using CSV, DataFrames, Printf

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
        "softmax" => 6
    )[x[2]]

extract_adjustment(x) = Dict(
        "QuaPy" => "",
        nothing => "un-adjusted (Eq.~\\ref{eq:binary-cc} / Eq.~\\ref{eq:binary-pcc})",
        "ovr" => "one-vs-rest (Eq.~\\ref{eq:ovr})",
        "inv" => "inverse (Eq.~\\ref{eq:inv})",
        "pinv" => "pseudo-inverse (Eq.~\\ref{eq:pinv})",
        "constrained" => "constrained (Eq.~\\ref{eq:constrained})",
        "softmax" => "soft-max (Eq.~\\ref{eq:softmax})"
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

function table_for_metric(df, metric)
    df = select_best(df, metric)
    @info "Selected the best methods for the metric $(metric)" best=df
    matches = match.(r"(\w+)(?: \((\w+)\))?", df[!,:method]) # separate method from adjustment
    df[!,:adjustment] = extract_adjustment.(matches)
    df[!,:order] = extract_order.(matches)
    df[!,:method] = extract_method.(matches)
    df = sort(df[df[!,:order] .> 0,:], :order) # remove reference methods
    df = unstack(df, :adjustment, :method, metric)
    df[!,"ACC / CC"] = format_scores(df[!,"ACC / CC"])
    df[!,"PACC / PCC"] = format_scores(df[!,"PACC / PCC"])
    return df
end

function export_table(output_path, df)
    open(output_path, "w") do io
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
    @info "LaTeX table exported to $(output_path)"
    return nothing
end

df = CSV.read("results/lequa_validation.csv", DataFrame)
@info "Configurations with failures" df[(df[!,:n_pdf_failures].>0) .| (df[!,:n_failures].>0),:]
df = innerjoin(
    table_for_metric(df, :ae),
    table_for_metric(df, :rae),
    on = :adjustment,
    makeunique = true,
    renamecols = Pair("; AE", "; RAE")
)
export_table("results/lequa_validation.tex", df[!,[1,2,4,3,5]]) # re-order columns
