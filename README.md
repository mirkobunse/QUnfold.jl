[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mirkobunse.github.io/QUnfold.jl/dev)
[![Build Status](https://github.com/mirkobunse/QUnfold.jl/workflows/CI/badge.svg)](https://github.com/mirkobunse/QUnfold.jl/actions)

# QUnfold.jl - algorithms for quantification and unfolding

This Julia package implements our unified framework of quantification and unfolding algorithms.


## Quick start

A detailed [documentation](https://mirkobunse.github.io/QUnfold.jl/dev) is available online.

```julia
using QUnfold, ScikitLearn
@sk_import ensemble: RandomForestClassifier

# X_trn, y_trn = my_training_data(...)

acc = ACC( # a scikit-learn bagging classifier with oob_score is needed
    RandomForestClassifier(oob_score=true)
)
trained_acc = fit(acc, X_trn, y_trn) # fit returns a trained COPY

# X_tst = my_testing_data(...)

p_est = predict(trained_acc, X_tst) # return a prevalence vector
```
