# Computing the acceptance correction factors for arbitrary binnings

Get the gamma test samples of the FACT data and the CORSIKA headers from which this data is derived. You require SSH access to the system of the e5b astro-particle physics group.

```
rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/build/apa85/gamma_test_dl3.hdf5 .

rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/data/corsika_headers/gamma_headers_corsika76900.hdf5 .
```

Our script has the following software requirements.

- Python 3.10
- HDF5 (system-wide installation, e.g., `apt-get install libhdf5-dev`)

Install all Python package dependencies in a virtual environment.

```
python -m venv venv
venv/bin/pip install --no-binary=tables --no-binary=h5py https://github.com/fact-project/fact_funfolding/archive/v0.3.6.tar.gz
```

The script `generate_acceptance.py` then produces a CSV file with acceptance correction factors.

```
venv/bin/python generate_acceptance.py gamma_test_dl3.hdf5 gamma_headers_corsika76900.hdf5 acceptance.csv
```

# Extracting the Training Data

Get the training samples and extract a CSV from it.

```
rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/build/apa85/gamma_train.hdf5 .

julia --project=../.. -e 'include("generate_training.jl"); process_training()'
```

