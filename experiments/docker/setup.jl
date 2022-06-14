using Pkg
@info "Installing package dependencies with setup.jl"
pkg"instantiate"
pkg"build"
pkg"precompile"

# fix Python package versions for reproducibility
using PyCall
pyimport("pip").main(["install", "scikit-learn==1.1.1", "quapy==0.1.6"])
