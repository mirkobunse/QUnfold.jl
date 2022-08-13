using CSV, DataFrames, HDF5, H5Zblosc

function download_crab()
    dl2_url = "https://factdata.app.tu-dortmund.de/dl2/FACT-Tools/v1.1.2/open_crab_sample_facttools_dl2.hdf5"
    dl2_path = "open_crab_sample_facttools_dl2.hdf5"
    dl3_url = "https://factdata.app.tu-dortmund.de/dl3/FACT-Tools/v1.1.2/open_crab_sample_dl3.hdf5"
    dl3_path = "open_crab_sample_dl3.hdf5"
    for (url, path) in [ (dl2_url, dl2_path), (dl3_url, dl3_path) ]
        @info "Downloading $(url) to $(path)"
        Base.download(url, path)
    end
end

# sub-routine creating a meaningful DataFrame from a full HDF5 file
function process_crab(
        output_path::String = "crab.csv";
        dl2_path::String = "open_crab_sample_facttools_dl2.hdf5",
        dl3_path::String = "open_crab_sample_dl3.hdf5",
        theta_cut::Float64 = sqrt(0.025),
        prediction_threshold::Float64 = 0.85
        )
    df = DataFrame()

    # read each HDF5 groups that astro-particle physicists read according to
    # https://github.com/fact-project/open_crab_sample_analysis/blob/f40c4fab57a90ee589ec98f5fe3fdf38e93958bf/configs/aict.yaml#L30
    features = [
        :size,
        :width,
        :length,
        :skewness_trans,
        :skewness_long,
        :concentration_cog,
        :concentration_core,
        :concentration_one_pixel,
        :concentration_two_pixel,
        :leakage1,
        :leakage2,
        :num_islands,
        :num_pixel_in_shower,
        :photoncharge_shower_mean,
        :photoncharge_shower_variance,
        :photoncharge_shower_max,
    ]
    id = [
        :run_id,
        :event_num,
        :night
    ]
    @info "Reading $(dl2_path)"
    h5open(dl2_path, "r") do file
        for f in [
                :cog_x, # needed only for feature generation; will be removed
                :cog_y,
                id..., # needed only for joining; will be removed
                features...,
                ]
            df[!, f] = read(file, "events/$(f)")
        end
    end

    # generate additional features, according to
    # https://github.com/fact-project/open_crab_sample_analysis/blob/f40c4fab57a90ee589ec98f5fe3fdf38e93958bf/configs/aict.yaml#L50
    df[!, :log_size] = log.(df[!, :size])
    df[!, :area] = df[!, :width] .* df[!, :length] .* π
    df[!, :size_area] = df[!, :size] ./ df[!, :area]
    df[!, :cog_r] = sqrt.(df[!, :cog_x].^2 + df[!, :cog_y].^2)

    # read with DL3
    dl3 = DataFrame()
    @info "Reading $(dl3_path)"
    h5open(dl3_path, "r") do file
        for f in [
                id..., # needed for joining
                :gamma_prediction,
                :theta_deg,
                :theta_deg_off_1,
                :theta_deg_off_2,
                :theta_deg_off_3,
                :theta_deg_off_4,
                :theta_deg_off_5,
                ]
            dl3[!, f] = read(file, "events/$(f)")
        end
    end

    # apply cuts
    N_dl3 = nrow(dl3) # remember the full size for logging
    dl3 = dl3[dl3[!,:gamma_prediction] .>= prediction_threshold,:]
    N_gamma = nrow(dl3)
    dl3[!,:is_on] = dl3[!,:theta_deg] .< theta_cut
    dl3[!,:is_off] =
        (dl3[!,:theta_deg_off_1] .< theta_cut) .|
        (dl3[!,:theta_deg_off_2] .< theta_cut) .|
        (dl3[!,:theta_deg_off_3] .< theta_cut) .|
        (dl3[!,:theta_deg_off_4] .< theta_cut) .|
        (dl3[!,:theta_deg_off_5] .< theta_cut)
    dl3 = dl3[dl3[!,:is_on] .| dl3[!,:is_off],:]
    @info "Read $(N_dl3) DL3 events, $(N_gamma) gamma-like ones, $(sum(dl3[!,:is_on])) on, and $(sum(dl3[!,:is_off])) off"

    # join
    N_dl2 = nrow(df) # remember the full size for logging
    df = innerjoin(df, dl3, on=id)
    @info "Joined $(N_dl2) DL2 events into $(sum(df[!,:is_on])) on and $(sum(df[!,:is_off])) off events"

    # convert the features to Float32, to find non-finite elements
    is_on = df[!,:is_on] # remember; do not convert to Float32
    df = df[!, [:log_size, :area, :size_area, :cog_r, features...]]
    for column in names(df)
        df[!, column] = convert.(Float32, df[!, column])
    end
    df[!,:is_on] = is_on
    df = filter(row -> all([ isfinite(cell) for cell in row ]), df)

    # store and return
    @info "Writing to $(output_path)"
    CSV.write(output_path, df)
    return df
end
