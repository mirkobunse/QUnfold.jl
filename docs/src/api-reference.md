# API reference

Below, you find a listing of all public methods of this package. Any other method you might find in the source code is not intended for direct usage.

```@meta
CurrentModule = QUnfold
```


## Common interface

TODO with an exemplary link to [`fit`](@ref).

```@docs
fit
predict
predict_with_background
```


## Quantification / unfolding methods

### CC

```@docs
CC
```

### ACC

```@docs
ACC
```

### PCC

```@docs
PCC
```

### PACC

```@docs
PACC
```

### RUN

```@docs
RUN
```

### SVD

```@docs
SVD
```

### HDx

```@docs
HDx
```

### HDy

```@docs
HDy
```

### IBU

```@docs
IBU
```

### SLD

```@docs
SLD
```


## Feature transformations

The unfolding methods [`RUN`](@ref), [`SVD`](@ref), and [`IBU`](@ref) have the flexibility of choosing between different feature transformations.

```@docs
ClassTransformer
TreeTransformer
HistogramTransformer
```
