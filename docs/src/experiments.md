# Experiments

The `experiments/` directory contains a Makefile with which you can run the experiments for our publication [On Multi-Class Extensions of Adjusted Classify and Count](https://lq-2022.github.io/proceedings/CompleteVolume.pdf).

```
@InProceedings{bunse2022multiclass,
  author    = {Mirko Bunse},
  title     = {On Multi-Class Extensions of Adjusted Classify and Count},
  booktitle = {Int. Worksh. on Learn. to Quantify: Meth. and Appl.},
  year      = {2022},
  pages     = {43--50},
}
```

## Running the experiments

**CAUTION:** We have run these experiments on 40 cores with 48 GB of RAM; with this setup, the experiments took 67 h. If you just want to check whether the scripts work, you can call `make -f lequa.mk tests` to traverse the entire code path with just a few iterations; this test completes in a few minutes.

```
cd experiments/
make -f lequa.mk -n # inspect all steps of our experimentation without running them (dry-run)

make -f lequa.mk # run all experimentation (CAUTION: computation-heavy)
```

## Docker setup

We provide a [Docker](https://docs.docker.com/) setup for those who prefer to run the experiments in an isolated environment, and possibly in a computing cluster.

```
cd experiments/docker/
make # build the Docker image
./run.sh # start an interactive Docker container from the image
```
