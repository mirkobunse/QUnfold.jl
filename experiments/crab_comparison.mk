_JULIA=julia --project=. --threads=auto

experiment: results/crab_comparison_poisson.tex
results/crab_comparison_poisson.tex: crab_comparison.jl
	$(_JULIA) $< results/crab_comparison_app_oq.tex results/crab_comparison_npp_crab.tex $@

test: results/test_crab_comparison_poisson.tex
results/test_crab_comparison_poisson.tex: crab_comparison.jl
	$(_JULIA) $< --is_test_run --test_path results/test_crab_comparison_test.csv --validation_path results/test_crab_comparison_validation.csv results/test_crab_comparison_app_oq.tex results/test_crab_comparison_npp_crab.tex $@

.PHONY: experiment test
