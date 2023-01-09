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
DecisionTreeClassifier = pyimport_conda("sklearn.tree", "scikit-learn").DecisionTreeClassifier

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
    conf = conf.astype(np.float64)
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

    M_acc = QUnfold.fit(ACC(c; fit_classifier=false), X_trn, y_trn).M
    M_acc_quapy = py"accPteCondEstim"(1:3, y_trn, y_oob)
    @test M_acc ≈ M_acc_quapy

    M_pacc = QUnfold.fit(PACC(c; fit_classifier=false), X_trn, y_trn).M
    M_pacc_quapy = py"paccPteCondEstim"(1:3, y_trn, c.oob_decision_function_)

    # repeat the PACC test with a sparse scipy matrix
    X_sparse = pyimport_conda("scipy.sparse", "scipy").csc_matrix(X_trn)
    M_sparse = QUnfold.fit(PACC(c; fit_classifier=false), X_sparse, y_trn).M

    # repeat the PACC test with minimum(y) == 0
    M_zero = QUnfold.fit(PACC(c; fit_classifier=false), X_trn, y_trn .- 1).M
    @test M_pacc ≈ M_pacc_quapy
    @test M_pacc ≈ M_sparse
    @test M_pacc ≈ M_zero

    # call other transformers
    M_hist = QUnfold.fit(RUN(QUnfold.HistogramTransformer(2)), X_trn, y_trn).M
    @test size(M_hist) == (4, 3) # 4 = 2 bins * 2 features

    m = QUnfold.fit(RUN(QUnfold.TreeTransformer(DecisionTreeClassifier(max_leaf_nodes=9))), X_trn, y_trn)
    @test size(m.M) == (9, 3)
    @test mean(y_trn .== ScikitLearn.predict(m.f.tree, X_trn)) > 0.5
end # testset

Random.seed!(42)
c = RandomForestClassifier(; oob_score=true, random_state=rand(UInt32))
t = TreeTransformer(DecisionTreeClassifier(max_leaf_nodes=9, random_state=rand(UInt32)))
for (name, method) in [
        "o-ACC (constrained, τ=10.0)" => ACC(c; τ=10.0, strategy=:constrained),
        "ACC (constrained)" => ACC(c; strategy=:constrained),
        "ACC (softmax)" => ACC(c; strategy=:softmax),
        "ACC (softmax_reg)" => ACC(c; strategy=:softmax_reg),
        "ACC (softmax_full_reg)" => ACC(c; strategy=:softmax_full_reg),
        "ACC (pinv)" => ACC(c; strategy=:pinv),
        "CC" => CC(c),
        "o-PACC (constrained, τ=10.0)" => PACC(c; τ=10.0, strategy=:constrained),
        "PACC (constrained)" => PACC(c; strategy=:constrained),
        "PACC (softmax)" => PACC(c; strategy=:softmax),
        "PACC (softmax_reg)" => PACC(c; strategy=:softmax_reg),
        "PACC (softmax_full_reg)" => PACC(c; strategy=:softmax_full_reg),
        "PACC (pinv)" => PACC(c; strategy=:pinv),
        "PCC" => PCC(c),
        "RUN (constrained, τ=1e-6)" => RUN(t; strategy=:constrained, τ=1e-6),
        "RUN (softmax, τ=1e-6)" => RUN(t; strategy=:softmax, τ=1e-6),
        "RUN (softmax_reg, τ=1e-6)" => RUN(t; strategy=:softmax_reg, τ=1e-6),
        "RUN (softmax_full_reg, τ=1e-6)" => RUN(t; strategy=:softmax_full_reg, τ=1e-6),
        "RUN (constrained, τ=10.0)" => RUN(t; strategy=:constrained, τ=10.0),
        "RUN (softmax, τ=10.0)" => RUN(t; strategy=:softmax, τ=10.0),
        "RUN (softmax_reg, τ=10.0)" => RUN(t; strategy=:softmax_reg, τ=10.0),
        "RUN (softmax_full_reg, τ=10.0)" => RUN(t; strategy=:softmax_full_reg, τ=10.0),
        "SVD (constrained, τ=1e-6)" => QUnfold.SVD(t; strategy=:constrained, τ=1e-6),
        "SVD (softmax, τ=1e-6)" => QUnfold.SVD(t; strategy=:softmax, τ=1e-6),
        "SVD (softmax_reg, τ=1e-6)" => QUnfold.SVD(t; strategy=:softmax_reg, τ=1e-6),
        "SVD (softmax_full_reg, τ=1e-6)" => QUnfold.SVD(t; strategy=:softmax_full_reg, τ=1e-6),
        "SVD (constrained, τ=10.0)" => QUnfold.SVD(t; strategy=:constrained, τ=10.0),
        "SVD (softmax, τ=10.0)" => QUnfold.SVD(t; strategy=:softmax, τ=10.0),
        "SVD (softmax_reg, τ=10.0)" => QUnfold.SVD(t; strategy=:softmax_reg, τ=10.0),
        "SVD (softmax_full_reg, τ=10.0)" => QUnfold.SVD(t; strategy=:softmax_full_reg, τ=10.0),
        "o-HDx (constrained, τ=10.0)" => HDx(3; τ=10.0, strategy=:constrained),
        "HDx (softmax_reg)" => HDx(3; strategy=:softmax_reg),
        "HDx (softmax_full_reg)" => HDx(3; strategy=:softmax_full_reg),
        "HDx (constrained)" => HDx(15; strategy=:constrained),
        "HDx (softmax)" => HDx(3; strategy=:softmax),
        "o-HDy (constrained, τ=10.0)" => HDy(c, 3; τ=10.0, strategy=:constrained),
        "HDy (softmax_reg)" => HDy(c, 3; strategy=:softmax_reg),
        "HDy (softmax_full_reg)" => HDy(c, 3; strategy=:softmax_full_reg),
        "HDy (constrained)" => HDy(c, 3; strategy=:constrained),
        "HDy (softmax)" => HDy(c, 3; strategy=:softmax),
        "RUN (original, n_df=2)" => RUN(t; strategy=:original, n_df=2),
        "SVD (original, n_df=2)" => QUnfold.SVD(t; strategy=:original, n_df=2),
        "IBU (o=0, λ=1.)" => IBU(t; o=0, λ=1.),
        "o-SLD (o=0, λ=1.)" => SLD(c; o=0, λ=1.),
        ]
    @testset "$name" begin
        Random.seed!(42) # each method gets the same 10 trials
        mae = Float64[] # mean absolute error over all trials
        for trial_seed in rand(UInt32, 10)
            Random.seed!(trial_seed)
            M = diagm( # construct and artificial quantification task
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

            m = QUnfold.fit(method, X_trn, y_trn)
            m_zero = QUnfold.fit(method, X_trn, y_trn .- 1)
            @test m.M == m_zero.M
            q = mean(QUnfold._transform(m.f, X_tst), dims=1)[:]
            q_zero = mean(QUnfold._transform(m_zero.f, X_tst), dims=1)[:]
            @test q == q_zero
            p_hat = QUnfold.predict(m, X_tst)
            @debug name p_hat
            @test p_hat ≈ QUnfold.predict(m_zero, X_tst) atol=0.1
            push!(mae, sum(abs.(p_hat .- p_tst)))
        end
        @info "MAE of $name: $(mean(mae))"
    end # testset
end # for (name, method)
