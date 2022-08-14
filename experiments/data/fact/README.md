# Computing the acceptance correction factors for arbitrary binnings

For getting the CORSIKA headers, you require SSH access to the system of the e5b astro-particle physics group.

```
rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/data/corsika_headers/gamma_headers_corsika76900.hdf5 .
```

Our script has the following dependencies.

- Python 3.10
- HDF5 (system-wide installation, e.g., `apt-get install libhdf5-dev`)

Install all required Python packages in a virtual environment.

```
python -m venv venv
venv/bin/pip install --no-binary=tables --no-binary=h5py https://github.com/fact-project/fact_funfolding/archive/v0.3.6.tar.gz
```

The script `generate_acceptance.py` then produces a CSV file with acceptance correction factors.

For this step, you need a file `gamma_test_dl3.hdf5`, which we discuss in the following sections

```
venv/bin/python generate_acceptance.py gamma_test_dl3.hdf5 gamma_headers_corsika76900.hdf5 acceptance.csv
```

## Data source, variant A: open Crab data

You can get a `gamma_test_dl3.hdf5` from the open Crab sample analysis: https://github.com/fact-project/open_crab_sample_analysis

You will also need a `gamma_train.hdf5` and a `gamma_test.hdf5`, which are also provided by this analysis.

```
export OPEN_CRAB_SAMPLE_ANALYSIS="path/to/open_crab_sample_analysis/build"
ln -s ${OPEN_CRAB_SAMPLE_ANALYSIS}/gamma_test_dl3.hdf5 .
ln -s ${OPEN_CRAB_SAMPLE_ANALYSIS}/gamma_train.hdf5 .
ln -s ${OPEN_CRAB_SAMPLE_ANALYSIS}/gamma_test.hdf5 .
```

## Data source, variant B: closed data of Max Nöthe's PhD thesis

You can get another `gamma_test_dl3.hdf5` and `gamma_train.hdf5` from Max Nöthe's thesis.

```
rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/build/apa85/gamma_test_dl3.hdf5 gamma_test_dl3_mnoethe.hdf5

rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/build/apa85/gamma_train.hdf5 gamma_train_mnoethe.hdf5

rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/build/apa85/gamma_test.hdf5 gamma_test_mnoethe.hdf5
```
