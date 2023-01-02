_TABLES=results/lequa.tex results/lequa_validation.tex
_EXPERIMENTS=results/lequa.csv results/lequa_validation.csv
_TESTS=results/test_lequa.tex
_DATA=data/T1B/public/training_data.txt data/T1B/public/test_prevalences.txt data/T1B/public/test_samples/4999.txt
_TRAINING_URL=https://zenodo.org/record/6546188/files/T1B.train_dev.zip?download=1
_PREVALENCES_URL=https://zenodo.org/record/6546188/files/T1B.test_prevalences.zip?download=1
_TEST_URL=https://zenodo.org/record/6546188/files/T1B.test.zip?download=1

tables: $(_TABLES)
results/lequa.tex: generate_lequa_tables.jl results/lequa_validation.csv results/lequa.csv
	julia --project=. $^ $@
results/lequa_validation.tex: generate_lequa_tables.jl results/lequa_validation.csv
	julia --project=. $^ results/lequa_validation.csv $@

experiments: $(_EXPERIMENTS)
results/lequa.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< $@
results/lequa_validation.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< --is_validation_run $@

tests: $(_TESTS)
results/test_lequa.tex: generate_lequa_tables.jl results/test_lequa_validation.csv results/test_lequa.csv
	julia --project=. $^ $@
results/test_lequa.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< --configuration dev --is_test_run --n_samples 3 $@
results/test_lequa_validation.csv: lequa.jl $(_DATA) $(shell find ../src -name "*.jl")
	julia --project=. $< --configuration dev --is_validation_run --is_test_run --n_samples 3 $@

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

.PHONY: tables experiments tests data
