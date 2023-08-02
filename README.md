# Using this repository

## This repository implements the download of FRED dataset described in this paper

If you use this dataset or code in any context, please cite this:
```
@article{oreshkin2021zero, 
	title={Meta-Learning Framework with Applications to Zero-Shot Time-Series Forecasting}, 
	volume={35},
	number={10}, 
	journal={in Proc. AAAI}, 
	author={Oreshkin, Boris N. and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua},
	year={2021}, 
	pages={9242--9250} 
}
```

## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone https://github.com/boreshkinai/fred_timeseries_dataset```

## Build docker image and launch container 
```
docker build -f Dockerfile -t fred_timeseries_dataset:$USER .
nvidia-docker run -p 8888:8888 -p 6006:6006 -v ~/workspace/fred_timeseries_dataset:/workspace/fred_timeseries_dataset -t -d --shm-size="1g" --name fred_timeseries_dataset_$USER fred_timeseries_dataset:$USER
```

### Download datasets

This project is based on M3 and M4 competition datasets. The following downloads and unpacks them

```bash
make init-datasets
```

### Build experiments

```bash
make build experiment=experiments/experiment_01 name=experiment_name
```

### Run experiments

```bash
make run experiment=experiments/experiment_01 name=experiment_name
```

### Build and run

```bash
make build run experiment=experiments/experiment_01 name=experiment_name
```

### Experiment summary
```bash
make summary experiment=experiments/experiment_01 name=experiment_name filter=*
```

### JupyterLab

```bash
make jupyterlab port=port gpu=gpu-id
```

### Tensorboard

TODO

### Custom command

It is possible to run any command within the docker.

```bash
make exec cmd='any command with --parameters'
```

## Development Convention

1. Create experiment package in `experiments`. Hierarchy of experiments is supported.
2. Create main.py with 3 methods: `init(name: str)`, `run` and `summary(name: str)`. 
For example, if experiment in package `experiment_01`:
```python
import fire

from common.experiment import create_experiment, load_experiment_parameters

from experiments.test.parameters import parameters

module_path = 'experiments/experiment_01'  # must be the same as packages from source.

def init(name: str):
    create_experiment(experiment_path=f'/project/{module_path}',
                      parameters=parameters[name],
                      command=lambda path, params: f'python {module_path}/main.py run --path="{path}"')

def run(path: str):
    load_experiment_parameters(path)
    # experiment logic
    

def summary(name: str):
    pass

if __name__ == '__main__':    
    fire.Fire()
```
Note: you can control what is passed to `run` (and you can rename it too).

For parameters, a good practice is to create `parameters.py` with a dictionary with sub-experiment name as key and 
dictionary with experiment parameters as values.

Also it may worth to extract model logic and dataset logic to separate modules. Thus, a typical experiment would contain
`main.py`, `parameters.py`, `model.py`, `dataset.py` and a notebook if necessary.


## Run Tests
```bash
make test
```
