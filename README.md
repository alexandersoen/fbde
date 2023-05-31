# Code SI
General code for "Fair Densities via Boosting the Sufficient Statistics of Exponential Families".

## Installation
We configure using conda `environment.yml` provides dependencies to run the code.
Additional files need to be setup and downloaded for AIF360 datasets to work.

## Demo
`run_discrete.py` runs FBDE for discrete datasets. An example:

```
ipython run_discrete.py -- --equal-rr --boosting-steps 32 --max-depth 8 --sattr 'sex' --seed 1 --dataset 'compas' --sr 0.8 --leverage 'exact'
```

Alternatively, one can run the continuous dataset code with `run_continuous.py`:

```
ipython run_continuous.py -- --equal-rr --boosting-steps 8 --max-depth 8  --seed 1 --dataset 'minneapolis' --sr 0.8 --leverage 'exact'
```

Additionally, we provide `*_data.py` scripts to run evaluation on the raw data.

## Datasets

One can use `compas` and `adult` for `--dataset` options, with `sex` or `race` as `--sattr`.
Otherwise, one can use `german` and `dutch` for `--dataset` options, with `sex` or `age` as `--sattr`.

## Citation

*Fair Densities via Boosting the Sufficient Statistics of Exponential Families* <br>
Alexander Soen, Hisham Husain, Richard Nock <br>
International Conference on Machine Learning, ICML 2023

**Please cite the corresponding paper when using the code**
