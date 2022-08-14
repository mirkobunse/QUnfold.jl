_JULIA=julia --project=. --threads=auto

experiments: results/crab_comparison_01k.tex results/crab_comparison_10k.tex
results/crab_comparison_01k.tex: crab_comparison.jl
	$(_JULIA) $< --test_path results/crab_comparison_01k_test.csv --validation_path results/crab_comparison_01k_validation.csv --N 1000 $@
results/crab_comparison_10k.tex: crab_comparison.jl
	$(_JULIA) $< --test_path results/crab_comparison_10k_test.csv --validation_path results/crab_comparison_10k_validation.csv --N 10000 $@

tests: results/test_crab_comparison.tex
results/test_crab_comparison.tex: crab_comparison.jl
	$(_JULIA) $< --test_path results/test_crab_comparison_test.csv --validation_path results/test_crab_comparison_validation.csv --is_test_run --N 876 $@

results/crab_protocols_nmd.tex: crab_protocols.jl
	julia --project=. $^ results/crab_protocols_Tp.tex $@

.PHONY: experiments tests
