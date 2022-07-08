# Computing the acceptance correction factors for arbitrary binnings

Get the gamma test samples of the open FACT data

```
wget https://factdata.app.tu-dortmund.de/dl3/FACT-Tools/v1.1.2/gamma_test_dl3.hdf5
```

and the CORSIKA headers from which this data is derived. For this second step, your require SSH access to the system of the e5b astro-particle physics group.

```
rsync --info=progress2 --human-readable phobos:/net/big-tank/POOL/users/mnoethe/phd_thesis/data/corsika_headers/gamma_headers_corsika76900.hdf5 .
```

Moreover, install all Python package dependencies in a virtual environment.

```
python -m venv venv
venv/bin/pip install https://github.com/fact-project/fact_funfolding/archive/v0.3.6.tar.gz
```

The script `generate_acceptance.py` then produces a CSV file with acceptance correction factors.

```
venv/bin/python generate_acceptance_correction.py gamma_test_dl3.hdf5 gamma_headers_corsika76900.hdf5 acceptance.csv
```
