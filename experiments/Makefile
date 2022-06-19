_RESULTS=results/lequa.csv results/lequa_validation.csv
_TESTS=results/test_lequa.csv
_DATA=data/T1B/public/training_data.txt data/T1B/public/test_prevalences.txt data/T1B/public/test_samples/4999.txt
_TRAINING_URL=https://zenodo.org/record/6546188/files/T1B.train_dev.zip?download=1
_PREVALENCES_URL=https://zenodo.org/record/6546188/files/T1B.test_prevalences.zip?download=1
_TEST_URL=https://zenodo.org/record/6546188/files/T1B.test.zip?download=1

results: $(_RESULTS)
results/lequa.csv: lequa.jl results/lequa_validation.csv $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< --methods_from_validation results/lequa_validation.csv $@
results/lequa_validation.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< --is_validation_run $@

tests: $(_TESTS)
results/test_lequa.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< --is_validation_run --is_test_run $@

data: $(_DATA)
data/T1B/public/training_data.txt: data/train_dev.zip
	7z x -o"data" $< && touch --no-create $@
data/T1B/public/test_prevalences.txt: data/test_prevalences.zip
	7z x -o"data" $< && touch --no-create $@
data/T1B/public/test_samples/4999.txt: data/test.zip
	7z x -o"data" $< && touch --no-create $@
data/train_dev.zip:
	curl --fail --create-dirs --output $@ $(_TRAINING_URL)
data/test_prevalences.zip:
	curl --fail --create-dirs --output $@ $(_PREVALENCES_URL)
data/test.zip:
	curl --fail --create-dirs --output $@ $(_TEST_URL)

.PHONY: results tests data
