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

function generate_data(p, M; n_samples=1000)
    y = StatsBase.sample(1:3, Weights(p), n_samples)
    X = zeros(n_samples, 2)
    for c in 1:3
        X[y.==c, 1] = StatsBase.sample(1:3, Weights(M[c,:]), sum(y.==c))
        X[y.==c, 2] = StatsBase.sample(1:3, Weights(M[c,:]), sum(y.==c))
    end
    X += rand(size(X)...) .* .1
    return X, y
end

@testset "ACC / PACC" begin
    M = diagm(
        0 => rand(3) * .7 .+ .1,
        1 => rand(2) * .05 .+ .025,
        -1 => rand(2) * .05 .+ .025,
        2 => rand(1) * .025,
        -2 => rand(1) * .025
    )
    M ./= sum(M, dims=2)
    X_trn, y_trn = generate_data([1, 2, 3], M)
    X_tst, y_tst = generate_data([3, 2, 1], M)
    p_tst = StatsBase.fit(Histogram, y_tst, 1:4).weights / length(y_tst)
    @info "Artificial quantification task " M p_tst

    c = RandomForestClassifier(; oob_score=true, random_state=rand(UInt32))
    ScikitLearn.fit!(c, X_trn, y_trn)
    for (name, method) in [
            "ACC (constrained)" => ACC(c; fit_classifier=false),
            "ACC (softmax)" => ACC(c; strategy=:softmax, fit_classifier=false),
            "ACC (pinv)" => ACC(c; strategy=:pinv, fit_classifier=false),
            "ACC (inv)" => ACC(c; strategy=:inv, fit_classifier=false),
            "CC" => CC(c; fit_classifier=false),
            "PACC (constrained)" => PACC(c; fit_classifier=false),
            "PACC (softmax)" => PACC(c; strategy=:softmax, fit_classifier=false),
            "PACC (pinv)" => PACC(c; strategy=:pinv, fit_classifier=false),
            "PACC (inv)" => PACC(c; strategy=:inv, fit_classifier=false),
            "PCC" => PCC(c; fit_classifier=false),
        ]
        @info name p_hat=QUnfold.predict(QUnfold.fit(method, X_trn, y_trn), X_tst)
    end

    # @test x_data[1] == x_data[2]
end # testset
