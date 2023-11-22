# synthcity-benchmark

This is the repository for "Synthcity: a benchmark framework for diverse use cases of tabular synthetic data".
However, this code just runs a simple pipeline taking advantage of the `benchmarks` available in `synthcity`. For 
more info on synthcity, see the [github](https://github.com/vanderschaarlab/synthcity) and [docs](https://synthcity.readthedocs.io/en/latest/metrics.html).

## :rocket: Installation

Install with:
```
pip install -r prereqs.txt
pip install -r requirements.txt
```

### Results

To reproduce the results from the paper run:

```
bash run_open_ml.sh
```

The adversarial random forest results can be generated seperately using:

```
bash run_open_ml_arf.sh
```