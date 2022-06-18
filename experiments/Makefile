_RESULTS=results/lequa_all.csv
_TESTS=results/test_lequa_all.csv
_DATA=data/T1B/public/training_data.txt
_URL=https://zenodo.org/record/6546188/files/T1B.train_dev.zip?download=1

results: $(_RESULTS)
results/lequa_all.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< results/lequa_best.csv $@

tests: $(_TESTS)
results/test_lequa_all.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< --is_test_run results/test_lequa_best.csv $@

data: $(_DATA)
data/T1B/public/training_data.txt: data/T1B.zip
	7z x -o"data" $<
data/T1B.zip:
	curl --fail --create-dirs --output $@ $(_URL)

.PHONY: results tests data
