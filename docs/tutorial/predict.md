# Prediction 

We describe here the SALTED functionalities to predict the electron density starting from a pretrained model. 

## Predict over a predefined set of structures 

The `inp.prediction` input section can be used to specify the filename (`inp.prediction.filename`) of a set of configurations for which we want to evaluate a given pretrained SALTED model. A `inp.prediction.predname` must also be specifyied to label the directories in which the predicted density coefficients and derived properties will be saved. It is then enough to run: 

```bash
python -m salted.prediction 
```

MPI parallelization can here be applied over configurations.

## Realtime prediction for a single structure

Realtime predictions for a single structure can be performed via the `salted.salted_prediction` function once a pretrained model is loaded via `salted.init_pred`. An example script which can be MPI-parallelized over atoms of the structure at hand can be found in `example/water_monomer_PySCF/test_prediction.py`. Depending on the specific application, an integer `lcut` value can be set to be smaller than the maximum angular momentum included in the basis set in order to truncate the predicted density coefficients accordingly. Moreover, the logical argument `gradient=True` can be used to enable computing the analytical gradient of the predicted coefficients with respect to atomic positions.
