import ScikitLearnBase

# general API

abstract type AbstractTransformer end
abstract type FittedTransformer end

"""
    _fit_transform(t, X, y) -> (f, fX, fy)

Return a copy `f` of the transformer `t` that is fitted to the data set `(X, y)`.
Also return the transformation of `X` and the sample of `y` which has been left
out for the computation of the transfer matrix `M`.
"""
_fit_transform(t::AbstractTransformer, X::Any, y::AbstractVector{T}) where {T <: Integer} =
    error("_fit_transform not implemented for $(typeof(m))")

"""
_transform(f, X) -> f(X)

Transform the data set `X` with the fitted transformer `f`.
"""
_transform(f::FittedTransformer, X::Any) =
    error("_transform not implemented for $(typeof(f))")

"""
_n_classes(f) -> Int

Return the number of classes known to the fitted transformer `f`.
"""
_n_classes(f::FittedTransformer) =
    error("_n_classes not implemented for $(typeof(f))")


# utility methods

"""
    onehot_encoding(y[, classes])

Map an array of labels / crisp predictions `y` to a one-hot encoded indicator matrix.
"""
onehot_encoding(y, classes=unique(y)) = Float64.(permutedims(classes) .== y)


# classification-based feature transformation

"""
    ClassTransformer(classifier; kwargs...)

This transformer yields the classification-based feature transformation used in `ACC`, `PACC`, `CC`, `PCC`, and `SLD`.

**Keyword arguments**

- `is_probabilistic = false` whether or not to use posterior predictions.
- `fit_classifier = true` whether or not to fit the given `classifier`.
"""
struct ClassTransformer <: AbstractTransformer
    classifier::Any
    is_probabilistic::Bool
    fit_classifier::Bool
    ClassTransformer(classifier::Any; is_probabilistic::Bool=false, fit_classifier::Bool=true) =
        new(classifier, is_probabilistic, fit_classifier)
end

struct FittedClassTransformer <: FittedTransformer
    classifier::Any
    is_probabilistic::Bool
end

function _fit_transform(t::ClassTransformer, X::Any, y::AbstractVector{T}) where {T<:Integer}
    classifier = t.classifier
    if minimum(y) ∉ [0, 1]
        @error "minimum(y) ∉ [0, 1]"
    end
    if !hasproperty(classifier, :oob_score) || !classifier.oob_score
        error("Only bagging classifiers with oob_score=true are supported")
    end # TODO add support for non-bagging classifiers
    if t.fit_classifier
        classifier = ScikitLearnBase.clone(classifier)
        ScikitLearnBase.fit!(classifier, X, y)
    end
    fX = classifier.oob_decision_function_
    is_finite = [ all(isfinite.(x)) for x in eachrow(fX) ] # Boolean vector
    fX = fX[is_finite,:]
    y = y[is_finite] .+ (1 - minimum(y)) # map to one-based labels
    if !t.is_probabilistic
        fX = onehot_encoding(
            mapslices(argmax, fX; dims=2)[:], # y_pred
            1:length(ScikitLearnBase.get_classes(classifier))
        )
    end
    return FittedClassTransformer(classifier, t.is_probabilistic), fX, y
end

function _transform(f::FittedClassTransformer, X::Any)
    fX = ScikitLearnBase.predict_proba(f.classifier, X)
    if f.is_probabilistic
        return fX
    else
        return onehot_encoding(
            mapslices(argmax, fX; dims=2)[:], # y_pred
            1:length(ScikitLearnBase.get_classes(f.classifier))
        )
    end
end

_n_classes(f::FittedClassTransformer) =
    length(ScikitLearnBase.get_classes(f.classifier))

# histogram-based feature transformation

"""
    HistogramTransformer(n_bins; kwargs...)

This transformer yields the histogram-based feature transformation used in `HDx` and `HDy`. The parameter `n_bins` specifies the number of bins *per input feature*.

**Keyword arguments**

- `preprocessor = nothing` can be another `AbstractTransformer` that is called before this transformer.
"""
struct HistogramTransformer <: AbstractTransformer
    n_bins::Int
    preprocessor::Union{AbstractTransformer,Nothing}
    HistogramTransformer(n_bins::Int; preprocessor::Union{AbstractTransformer,Nothing}=nothing) =
        new(n_bins, preprocessor)
end

struct FittedHistogramTransformer <: FittedTransformer
    edges::Matrix{Float64} # shape (n_bins-1, n_features)
    n_classes::Int
    preprocessor::Union{FittedTransformer,Nothing}
end

function _fit_transform(t::HistogramTransformer, X::Any, y::AbstractVector{T}) where {T<:Integer}
    n_classes = length(unique(y))
    preprocessor, X, y = _fit_transform(t.preprocessor, X, y)
    f = FittedHistogramTransformer(
        hcat(_edges.(eachcol(X), t.n_bins)...),
        n_classes,
        preprocessor
    )
    return f, _transform(f, X; apply_preprocessor=false), y .+ (1 - minimum(y))
end

_edges(x::AbstractVector{T}, n_bins::Int) where {T<:Real} =
    collect(1:(n_bins-1)) .* (maximum(x) - minimum(x)) / n_bins .+ minimum(x)

function _transform(f::FittedHistogramTransformer, X::AbstractArray; apply_preprocessor::Bool=true)
    if apply_preprocessor
        X = _transform(f.preprocessor, X)
    end
    n_bins = size(f.edges, 1) + 1
    fX = zeros(Int, size(X, 1), n_bins * size(X, 2))
    for j in 1:size(X, 2) # feature index
        edges = f.edges[:,j]
        offset = (j-1) * n_bins
        for i in 1:size(X, 1) # sample index
            fX[i, offset + searchsortedfirst(edges, X[i,j])] = 1
        end
    end
    return fX
end

_n_classes(f::FittedHistogramTransformer) = f.n_classes

_fit_transform(t::Nothing, X::Any, y::AbstractVector{T}) where {T<:Integer} = t, X, y
_transform(f::Nothing, X::Any) = X


# tree-induced partitioning

"""
    TreeTransformer(tree; kwargs...)

This transformer yields a tree-induced partitioning, as proposed by Börner et al., 2017: *Measurement/simulation mismatches and multivariate data discretization in the machine learning era*.

**Keyword arguments**

- `fit_frac = 1/5` is the fraction of data used for training the tree if `fit_tree==true`.
- `fit_tree = true` whether or not to fit the given `tree`.
"""
struct TreeTransformer <: AbstractTransformer
    tree::Any
    fit_tree::Bool
    fit_frac::Float64
    TreeTransformer(tree::Any; fit_tree::Bool=true, fit_frac::Float64=1/2) =
        new(tree, fit_tree, fit_frac)
end

struct FittedTreeTransformer <: FittedTransformer
    tree::Any
    index_map::Dict{Int,Int}
    x::Vector{Int} # optional hold-out data to return in _fit_transform
    y::Vector{Int}
end

function fit(t::TreeTransformer, X::AbstractArray, y::AbstractVector{T}) where {T<:Integer}
    tree = t.tree
    index_map = Dict{Int,Int}()
    x = Int[]
    if t.fit_tree
        tree = ScikitLearnBase.clone(tree)
        rng = MersenneTwister(tree.random_state)
        i_rand = randperm(rng, length(y)) # shuffle (X, y)
        i_tree = round(Int, length(y) * t.fit_frac) # where to split
        split_is_good = false
        for _ in 1:5 # attempt broken splits multiple times
            c_trn = sort(unique(y[i_rand[1:i_tree]]))
            c_val = sort(unique(y[i_rand[(i_tree+1):end]]))
            if length(c_trn) == length(c_val) && all(c_trn .== c_val)
                split_is_good = true
                break
            else
                @warn "Reattempting a split with missing labels" fit_frac c_trn c_val
                i_rand = randperm(rng, length(y))
            end
        end
        if !split_is_good
            error("Missing label in one of the splits with fit_frac=$(t.fit_frac): c_trn=$c_trn, c_val=$c_val")
        end
        ScikitLearnBase.fit!(tree, X[i_rand[1:i_tree], :], y[i_rand[1:i_tree]])

        # obtain all leaf indices by probing the tree with the training data
        x = _apply_tree(tree, X[i_rand[1:i_tree], :]) # leaf indices (rather arbitrary)
        index_map = Dict(zip(unique(x), 1:length(unique(x)))) # map to 1, …, F

        # limit (X, y) to the remaining data that was not used for fitting the tree
        x = _apply_tree(tree, X[i_rand[(i_tree+1):end], :]) # apply to the remaining data
        y = y[i_rand[(i_tree+1):end]]
    else
        # guess the leaf indices by probing the tree with the available data
        x = _apply_tree(tree, X)
        index_map = Dict(zip(unique(x), 1:length(unique(x))))
    end
    return FittedTreeTransformer(
        tree,
        index_map,
        [ index_map[x_i] for x_i ∈ x ], # map to 1, …, F
        y .+ (1 - minimum(y)) # map to one-based labels
    )
end

function _fit_transform(t::TreeTransformer, X::Any, y::AbstractVector{T}) where {T<:Integer}
    f = fit(t, X, y) # create a FittedTreeTransformer with hold-out data (x, y)
    fX = onehot_encoding(f.x, 1:length(f.index_map))
    return FittedTreeTransformer(f.tree, f.index_map, Int[], Int[]), fX, f.y # forget (x, y)
end

_fit_transform(f::FittedTreeTransformer, X::Any, y::AbstractVector{T}) where {T<:Integer} =
    (f, onehot_encoding(f.x, 1:length(f.index_map)), f.y) # assume hold-out data (x, y)

_transform(f::FittedTreeTransformer, X::Any) =
    onehot_encoding( # histogram representation
        [ f.index_map[x] for x ∈ _apply_tree(f.tree, X) ], # map to 1, …, F
        1:length(f.index_map)
    )

_n_classes(f::FittedTreeTransformer) =
    length(ScikitLearnBase.get_classes(f.tree))

_apply_tree(tree::Any, X::Any) = tree.apply(X)
