if ".." ∉ LOAD_PATH push!(LOAD_PATH, "..") end # add QUnfold to the LOAD_PATH
ENV["PYTHONWARNINGS"] = "ignore"
using
    CSV,
    DataFrames,
    StatsBase

function inspect(path="results/crab_comparison_01k_validation.csv")
    df = CSV.read(path, DataFrame)
    df[!,:exception] = coalesce.(df[!,:exception], "")

    for (group_key, group_df) ∈ pairs(groupby( # group average NMDs by method_id
                combine(
                    groupby(df[df[!,:exception].=="",:], [:protocol, :method_id, :method_name]),
                    :nmd => DataFrames.mean => :nmd
                ), # average NMDs
                [:protocol, :method_id]
            ))
        @info "$(group_key.protocol), method=$(group_key.method_id)"
        show(sort(group_df[:, [:method_name, :nmd]], :nmd), truncate=1000)
        println()
    end
end
