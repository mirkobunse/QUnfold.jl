# [QUnfold.jl](@id Home)

This Julia package implements our unified framework of quantification and unfolding algorithms.


## Installation

QUnfold.jl can be installed through the Julia package manager. From the Julia REPL, type `]` to enter the Pkg mode of the REPL. Then run

```
pkg> add QUnfold
```


## Quick start

Each quantification / unfolding technique implements a [`fit`](@ref) and a [`predict`](@ref) function.

- The [`fit`](@ref) function receives a training set `(X, y)` as an input. It returns a **trained copy** of the quantification / unfolding technique; no in-place training happens.
- The [`predict`](@ref) function receives a single sample of multiple data items. It returns the estimated vector of class prevalences within this sample.

The underlying classifier of each technique must be a bagging classifier with `oob_score=true`, which implements the API of [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl/).

```julia
using QUnfold, ScikitLearn
@sk_import ensemble: RandomForestClassifier

acc = ACC( # a scikit-learn bagging classifier with oob_score is needed
    RandomForestClassifier(oob_score=true)
)

# X_trn, y_trn = my_training_data(...)
trained_acc = fit(acc, X_trn, y_trn) # fit returns a trained COPY

# X_tst = my_testing_data(...)
p_est = predict(trained_acc, X_tst) # return a prevalence vector
```


## Methods

The following methods are implemented here:

- [`CC`](@ref): The basic Classify & Count method.
- [`ACC`](@ref): The Adjusted Classify & Count method.
- [`PCC`](@ref): The Probabilistic Classify & Count method.
- [`PACC`](@ref): The Probabilistic Adjusted Classify & Count method.
- [`RUN`](@ref): The Regularized Unfolding method.
- [`SVD`](@ref): The Singular Value Decomposition-based unfolding method.
- [`HDx`](@ref): The Hellinger Distance-based method on feature histograms.
- [`HDy`](@ref): The Hellinger Distance-based method on prediction histograms.
- [`IBU`](@ref): The Iterative Bayesian Unfolding method.
- [`SLD`](@ref): The Saerens-Latinne-Decaestecker method, a.k.a. EMQ or Expectation Maximization-based Quantification.

Most of these methods support regularization towards smooth estimates, which is beneficial in ordinal quantification.


## Citing

This implementation is a part of my Ph.D. thesis.

```
@PhdThesis{bunse2022machine,
  author = {Bunse, Mirko},
  school = {TU Dortmund University},
  title  = {Machine Learning for Acquiring Knowledge in Astro-Particle Physics},
  year   = {2022},
  doi    = {10.17877/DE290R-23021},
}
```
