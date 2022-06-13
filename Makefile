experiments/data/T1B/public/training_data.txt: experiments/data/T1B.zip
	7z x -o"experiments/data" $<
experiments/data/T1B.zip:
	curl --fail --create-dirs --output $@ https://zenodo.org/record/6546188/files/T1B.train_dev.zip?download=1
