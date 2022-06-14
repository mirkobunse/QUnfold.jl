_RESULTS=experiments/results/lequa_all.csv
_TESTS=experiments/results/test_lequa_all.csv
_DATA=experiments/data/T1B/public/training_data.txt

results: $(_RESULTS)
experiments/results/lequa_all.csv: experiments/lequa.jl $(_DATA)
	julia --project=experiments $< experiments/results/lequa_best.csv $@

tests: $(_TESTS)
experiments/results/test_lequa_all.csv: experiments/lequa.jl $(_DATA)
	julia --project=experiments $< --is_test_run experiments/results/test_lequa_best.csv $@

data: $(_DATA)
experiments/data/T1B/public/training_data.txt: experiments/data/T1B.zip
	7z x -o"experiments/data" $<
experiments/data/T1B.zip:
	curl --fail --create-dirs --output $@ https://zenodo.org/record/6546188/files/T1B.train_dev.zip?download=1

.PHONY: results tests data
