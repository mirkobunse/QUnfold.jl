import ScikitLearnBase


# general API

abstract type AbstractTransformer end
abstract type FittedTransformer end

"""
    _fit_transform(t, X, y) -> (f, fX, fy)

Return a copy `f` of the Transformer `t` that is fitted to the data set `(X, y)`.
Also return the transformation of `X` and the sample of `y` which has been left
out for the computation of the transfer matrix `M`.
"""
_fit_transform(t::AbstractTransformer, X::Any, y::AbstractVector{T}) where {T <: Integer} =
    error("_fit_transform not implemented for $(typeof(m))")

"""
_transform(f, X) -> f(X)

Transform the data set `X` with the transformer `f`.
"""
_transform(f::FittedTransformer, X::Any) =
    error("_transform not implemented for $(typeof(f))")


# utility methods

"""
    onehot_encoding(y[, classes])

Map an array of labels / crisp predictions `y` to a one-hot encoded indicator matrix.
"""
onehot_encoding(y, classes=unique(y)) = Float64.(permutedims(classes) .== y)

# classification-based feature transformation

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
    if !hasproperty(classifier, :oob_score) || !classifier.oob_score
        error("Only bagging classifiers with oob_score=true are supported")
    end # TODO add support for non-bagging classifiers
    if t.fit_classifier
        classifier = ScikitLearnBase.clone(classifier)
        ScikitLearnBase.fit!(classifier, X, y)
    end
    fX = classifier.oob_decision_function_
    i_finite = [ all(isfinite.(x)) for x in eachrow(fX) ]
    fX = fX[i_finite,:]
    y = y[i_finite]
    if !t.is_probabilistic
        fX = onehot_encoding(
            mapslices(argmax, fX; dims=2)[:], # y_pred
            ScikitLearnBase.get_classes(classifier)
        )
    end
    return FittedClassTransformer(classifier, t.is_probabilistic), fX, y
end

_transform(f::FittedClassTransformer, X::Any) =
    if f.is_probabilistic
        ScikitLearnBase.predict_proba(f.classifier, X)
    else
        onehot_encoding(
            ScikitLearnBase.predict(f.classifier, X),
            ScikitLearnBase.get_classes(f.classifier)
        )
    end
