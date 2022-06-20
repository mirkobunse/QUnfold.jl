
# QUnfold.jl - Algorithms for Quantification and Unfolding

This Julia package implements our unified framework of quantification and unfolding algorithms.

**Current Status:** At this point, we have only implemented several multi-class extensions of ACC. Other methods will follow shortly.


## Usage

Each quantification / unfolding technique implements a `fit` and a `predict` function.

- The `fit` function receives a training set `(X, y)` as an input. It returns a *trained copy* of the quantification / unfolding technique; no in-place training happens.
- The `predict` function receives a single sample of multiple data items. It returns the estimated vector of class prevalences within this sample.

The underlying classifier of each technique must implement the API of [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl/).

```julia
using QUnfold, ScikitLearn

@sk_import linear_model: LogisticRegression

# X_trn, y_trn = my_training_data(...)

acc = ACC(LogisticRegression())
trained_acc = fit(acc, X_trn, y_trn) # fit returns a trained COPY

# X_tst = my_testing_data(...)

p_est = predict(trained_acc, X_tst) # return a prevalence vector
```

Docstrings contain further information.


## Experiments

The `experiments/` directory contains a `Makefile` with which you can run the experiments for our LQ2022 submission *On Multi-Class Extensions of Adjusted Classify and Count*.

```
cd experiments/
make -n # inspect all steps of our experimentation without running them (dry-run)

make # run all experimentation (CAUTION: computation-heavy)
```

We provide a [Docker](https://docs.docker.com/) setup for those who prefer to run the experiments in an isolated environment, and possibly in a computing cluster.

```
cd experiments/docker/
make # build the Docker image
./run.sh # start an interactive Docker container from the image
```
