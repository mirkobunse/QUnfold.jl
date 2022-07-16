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

py"""
import numpy as np
from sklearn.metrics import confusion_matrix

# https://github.com/HLT-ISTI/QuaPy/blob/6a5c528154c2d6d38d9f3258e667727bf692fc8b/quapy/method/aggregative.py#L319
def accPteCondEstim(classes, y, y_):
    conf = confusion_matrix(y, y_, labels=classes).T
    conf = conf.astype(np.float)
    class_counts = conf.sum(axis=0)
    for i, _ in enumerate(classes):
        if class_counts[i] == 0:
            conf[i, i] = 1
        else:
            conf[:, i] /= class_counts[i]
    return conf

# https://github.com/HLT-ISTI/QuaPy/blob/6a5c528154c2d6d38d9f3258e667727bf692fc8b/quapy/method/aggregative.py#L450
def paccPteCondEstim(classes, y, y_):
    confusion = np.eye(len(classes))
    for i, class_ in enumerate(classes):
        idx = y == class_
        if idx.any():
            confusion[i] = y_[idx].mean(axis=0)
    return confusion.T
"""

@testset "Transfer matrix estimation" begin
    M_true = diagm(
        0 => rand(3) * .7 .+ .1,
        1 => rand(2) * .05 .+ .025,
        -1 => rand(2) * .05 .+ .025,
        2 => rand(1) * .025,
        -2 => rand(1) * .025
    )
    M_true ./= sum(M_true, dims=2)
    X_trn, y_trn = generate_data([1, 2, 3], M_true)
    c = RandomForestClassifier(; oob_score=true, random_state=rand(UInt32))
    ScikitLearn.fit!(c, X_trn, y_trn)
    y_oob = mapslices(argmax, c.oob_decision_function_; dims=2)[:]
    @info "OOB predictions" mean(y_trn .== y_oob)

    acc = ACC(c; fit_classifier=false)
    _, fX, fy = QUnfold._fit_transform(QUnfold._transformer(acc), X_trn, y_trn)
    M_acc = zeros(size(fX, 2), 3)
    for (fX_i, fy_i) in zip(eachrow(fX), fy)
        M_acc[:, fy_i] .+= fX_i
    end
    M_acc ./= sum(M_acc; dims=1)
    M_acc_quapy = py"accPteCondEstim"(1:3, y_trn, y_oob)
    @info "M_acc" M_acc M_acc_quapy

    pacc = PACC(c; fit_classifier=false)
    _, fX, fy = QUnfold._fit_transform(QUnfold._transformer(pacc), X_trn, y_trn)
    M_pacc = zeros(size(fX, 2), 3)
    for (fX_i, fy_i) in zip(eachrow(fX), fy)
        M_pacc[:, fy_i] .+= fX_i
    end
    M_pacc ./= sum(M_pacc; dims=1)
    M_pacc_quapy = py"paccPteCondEstim"(1:3, y_trn, c.oob_decision_function_)
    @info "M_pacc" M_pacc M_pacc_quapy
end # testset

@testset "Execution of all methods" begin
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
    t = QUnfold.fit(TreeTransformer(9), X_trn, y_trn)
    for (name, method) in [
            "o-ACC (constrained, τ=10.0)" => ACC(c; τ=10.0, strategy=:constrained, fit_classifier=false),
            "ACC (constrained)" => ACC(c; strategy=:constrained, fit_classifier=false),
            "ACC (softmax)" => ACC(c; strategy=:softmax, fit_classifier=false),
            "ACC (pinv)" => ACC(c; strategy=:pinv, fit_classifier=false),
            "ACC (inv)" => ACC(c; strategy=:inv, fit_classifier=false),
            "CC" => CC(c; fit_classifier=false),
            "o-PACC (constrained, τ=10.0)" => PACC(c; τ=10.0, strategy=:constrained, fit_classifier=false),
            "PACC (constrained)" => PACC(c; strategy=:constrained, fit_classifier=false),
            "PACC (softmax)" => PACC(c; strategy=:softmax, fit_classifier=false),
            "PACC (pinv)" => PACC(c; strategy=:pinv, fit_classifier=false),
            "PACC (inv)" => PACC(c; strategy=:inv, fit_classifier=false),
            "PCC" => PCC(c; fit_classifier=false),
            "RUN (constrained, τ=1e-6)" => RUN(t; strategy=:constrained, τ=1e-6),
            "RUN (softmax, τ=1e-6)" => RUN(t; strategy=:softmax, τ=1e-6),
            "RUN (constrained, τ=10.0)" => RUN(t; strategy=:constrained, τ=10.0),
            "RUN (softmax, τ=10.0)" => RUN(t; strategy=:softmax, τ=10.0),
            "SVD (constrained, τ=1e-6)" => QUnfold.SVD(t; strategy=:constrained, τ=1e-6),
            "SVD (softmax, τ=1e-6)" => QUnfold.SVD(t; strategy=:softmax, τ=1e-6),
            "SVD (constrained, τ=10.0)" => QUnfold.SVD(t; strategy=:constrained, τ=10.0),
            "SVD (softmax, τ=10.0)" => QUnfold.SVD(t; strategy=:softmax, τ=10.0),
            "o-HDx (constrained, τ=10.0)" => HDx(3; τ=10.0, strategy=:constrained),
            "HDx (constrained)" => HDx(15; strategy=:constrained),
            "HDx (softmax)" => HDx(3; strategy=:softmax),
            "o-HDy (constrained, τ=10.0)" => HDy(c, 3; τ=10.0, strategy=:constrained, fit_classifier=false),
            "HDy (constrained)" => HDy(c, 3; strategy=:constrained, fit_classifier=false),
            "HDy (softmax)" => HDy(c, 3; strategy=:softmax, fit_classifier=false),
            "RUN (original, n_df=2)" => RUN(t; strategy=:original, n_df=2),
            "SVD (original, n_df=2)" => QUnfold.SVD(t; strategy=:original, n_df=2),
            "IBU (o=2, λ=.5)" => IBU(t; o=2, λ=.5),
            "o-SLD (o=2, λ=.5)" => SLD(c; o=2, λ=.5),
            # "RUN (unconstrained, τ=10.0)" => RUN(t; strategy=:unconstrained, τ=10.0),
        ]
        @info name p_hat=QUnfold.predict(QUnfold.fit(method, X_trn, y_trn), X_tst)
    end

    # @test x_data[1] == x_data[2]
end # testset
