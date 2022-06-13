import ScikitLearnBase

abstract type AbstractTransformer end
abstract type FittedTransformer end


# general API

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


# classification-based feature transformation

struct ClassTransformer <: AbstractTransformer
    classifier::Any
    is_probabilistic::Bool
end

struct FittedClassTransformer <: FittedTransformer
    classifier::Any
    is_probabilistic::Bool
end

function _fit_transform(t::ClassTransformer, X::Any, y::AbstractVector{T}) where {T<:Integer}
    classifier = ScikitLearnBase.clone(t.classifier)
    if !hasproperty(classifier, :oob_score) || !classifier.oob_score
        error("Only bagging classifiers with oob_score=true are supported")
    end # TODO add support for non-bagging classifiers
    ScikitLearnBase.fit!(classifier, X, y)
    fX = if t.is_probabilistic
        classifier.oob_decision_function_
    else
        y_pred = mapslices(argmax, classifier.oob_decision_function_; dims=2)[:]
        permutedims(ScikitLearnBase.get_classes(classifier)) .== y_pred # one-hot encoding
    end
    return FittedClassTransformer(classifier, t.is_probabilistic), fX, y
end

_transform(f::FittedClassTransformer, X::Any) =
    if f.is_probabilistic
        ScikitLearnBase.predict_proba(f.classifier, X)
    else
        y_pred = ScikitLearnBase.predict(f.classifier, X)
        permutedims(ScikitLearnBase.get_classes(f.classifier)) .== y_pred # one-hot encoding
    end
