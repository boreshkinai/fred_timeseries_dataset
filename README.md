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

## Download FRED

Open the notebook and execute the cells. Don't forget to obtain and properly store the FRED API key as explained in the notebook

```bash
http://YOUR_SERVER_IP_ADDRESS:8888/notebooks/FredDownload.ipynb
```
