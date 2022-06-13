using
    LinearAlgebra,
    PyCall,
    QUnfold,
    Random,
    StatsBase,
    Test
import ScikitLearn

Random.seed!(42) # make tests reproducible

RandomForestClassifier = pyimport_conda("sklearn.ensemble", "scikit-learn").RandomForestClassifier

@testset "ACC / PACC" begin
    M = diagm(
        0 => rand(3) * .7,
        1 => rand(2) * .1,
        -1 => rand(2) * .1,
        2 => rand(1) * .05,
        -2 => rand(1) * .05
    )
    M ./= sum(M, dims=2)
    @show M
    y = rand(1:3, 100)
    X = zeros(length(y), 2)
    for c in 1:3
        X[y.==c, 1] = StatsBase.sample(1:3, Weights(M[c,:]), sum(y.==c))
        X[y.==c, 2] = StatsBase.sample(1:3, Weights(M[c,:]), sum(y.==c))
    end
    X += rand(size(X)...) .* .1
    @show X y

    acc = QUnfold.fit(ACC(RandomForestClassifier(; oob_score=true)), X, y)
    @show QUnfold.predict(acc, X)

    # @test x_data[1] == x_data[2]
end
