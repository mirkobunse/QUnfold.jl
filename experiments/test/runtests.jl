using
    LinearAlgebra,
    QUnfoldExperiments,
    Random,
    StatsBase,
    Test

Random.seed!(42) # make tests reproducible

ae(a, b) = mean(abs.(a - b))

@testset "Sampling" begin
    N = 1000000 # large samples for low errors
    avg_sim = mean(hcat(QUnfoldExperiments.sample_npp_simulation(N, 5)...); dims=2)[:]
    @test ae(avg_sim, QUnfoldExperiments.P_TRN[]) < 1e-6

    avg_crab = mean(hcat(QUnfoldExperiments.sample_npp_crab(N, 5)...); dims=2)[:]
    p_crab = QUnfoldExperiments.magic_crab_flux() .* QUnfoldExperiments.A_EFF[]
    @test ae(avg_crab, p_crab ./ sum(p_crab)) < 1e-6
    @test ae(avg_sim, p_crab ./ sum(p_crab)) > 1e-6 # cross-check
end # testset

@testset "Curvature" begin
    P = rand(5, 12)
    P ./= sum(P, dims=2)
    C = length(QUnfoldExperiments.BIN_CENTERS[])
    t = LinearAlgebra.diagm( # Tikhonov matrix
        -1 => fill(-1, C-1),
        0 => fill(2, C),
        1 => fill(-1, C-1)
    )[2:(C-1), :]
    for p ∈ eachrow(P)
        curvature_p = sum([ sum(t[i, j] * p[j] for j in 1:C)^2 for i in 1:(C-2) ]) / 2
        @test curvature_p ≈ QUnfoldExperiments.curvature(p)
    end
end # testset
