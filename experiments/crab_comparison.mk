_JULIA=julia --project=. --threads=auto

experiment: results/crab_comparison_10k.tex
results/crab_comparison_10k.tex: crab_comparison.jl
	$(_JULIA) $< results/crab_comparison_01k.tex $@

test: results/test_crab_comparison_10k.tex
results/test_crab_comparison_10k.tex: crab_comparison.jl
	$(_JULIA) $< --is_test_run --test_path results/test_crab_comparison_test.csv --validation_path results/test_crab_comparison_validation.csv results/test_crab_comparison_01k.tex $@

.PHONY: experiment test
