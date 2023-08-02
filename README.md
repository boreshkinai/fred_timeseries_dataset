### Initialize Project

```bash
$ mkdir -p <project-path>
$ cd <project-path>
$ git clone git@github.com:ElementAI/timeseries-tl.git source
```

All commands below should be executed from `<project-path>/source`

### Build docker

```bash
make init
```

This project requires NVidia GPU. But for some tests, where GPU is not required it's possible to pass `gpu=None` 
and start container without GPU.

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
