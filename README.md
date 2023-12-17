# :earth_africa: Nigeria Crop Mask :seedling:
This repository contains the code and data of [Gajardo et. al (2023)](arxiv link) for training a pixel-wise LSTM binary classifier to predict cropland vs non-cropland from remote sensing data and using it to generate two cropland maps for Nigeria for the year 2020. The code is largely based on the work of [Kerner et. al (2020)](https://arxiv.org/abs/2006.16866) from NASA Harvest, who build a similar cropland mask for Togo.

- :pencil: **[Paper](arxiv link):** Gajardo et. al (2023), *Country-scale Cropland Mapping in Data-Scarce Settings Using Deep Learning: a Case Study of Nigeria.*
- :clapper:**[Demo](https://nigeria-crop-mask.herokuapp.com/)**: an interactive visualization of the map and comparison to the ESA WorldCover 2020 land cover map can be found in Google Earth Engine (GEE).
- :link: **[Map visualization script](https://code.earthengine.google.com/22ca87f617ca91ca8c6e5176fa2466c2) (needs a GEE account)**


<p align="center">
    <img src="assets/nigeria_cropland_probability_map_image.png" alt="Nigeria map" height="600px"/>
</p>

## :open_file_folder: Data
The data used to train the LSTM model combines a hand-labelled dataset of crop and non-crop labels distributed throughout Nigeria (figure below) with a subset of the [global Geowiki cropland dataset](https://doi.pangaea.de/10.1594/PANGAEA.873912) to predict the presence of cropland in a pixel time series. The pixels time series consists of 12 monthly composites of remote sensing data at 10 m resolution, including Sentinel-1 and Sentinel-2 satellite images, as well as meteorological and topographic data. The training and inference data is processed using the [CropHarvest Python package](https://github.com/nasaharvest/cropharvest).

<p align="center">
    <img src="assets/nigeria_dataset_splits_new.png" alt="Nigeria map" height="500px"/>
</p>

## :hammer: Setup
The code was developed and tested on a Linux-based workstation using Python 3.7. For setting up the environment, install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or Anaconda and create a new environment:
Alternatively, you can install the explicit environment:
```bash
conda env create -f envs/env_gpu_explicit.yml
conda activate nigeria-crop-mask-gpu
```

If you run into package conflicts, you can try to install the dependencies manually:
```bash
conda env create -f envs/environment-gpu.yml # GPU environment
conda env create -f envs/environment.yml # CPU environment
```
### Google Earth Engine (optional)

We provide the labels in this repository [data/features/nigeria-cropharvest](data/features/nigeria-cropharvest/), as well as the respective data arrays in [google drive](link), but if you want to create the dataset yourself or download the inference data (uses 10TB of disk space though!) you will need a Google Earth Engine account. To use it, activate the conda environment, create a GEE acount and authenticate your account on the CL or in a Jupyter notebook:

```bash
earthengine authenticate # CL
python -c "import ee; ee.Authenticate()" # Jupyter notebook
```

Note that Google Earth Engine (GEE) exports files to Google Drive by default (to the same google account used to sign up to GEE).
Running exports can be viewed (and individually cancelled) in the `Tabs` bar on the [GEE Code Editor](https://code.earthengine.google.com/).

## :computer: Code

The main entrypoints into the pipeline are the [scripts](scripts) and the [notebooks](notebooks/). The `src` folder provides the implementation of the model, data exporters, utilities, etc. The `data` folder contains the raw data and the processed data. The `models` folder contains the trained models.

### Data preparation
* [scripts/export.py](scripts/export.py) exports data (locally, or to Google Drive - see below)
* [scripts/process.py](scripts/process.py) processes the raw data
* [scripts/engineer.py](scripts/engineer.py) combines the earth observation data with the labels to create (x, y) training data
   
### Training and testing
The scrip [scripts/models.py](scripts/models.py) is used to train models. The main results table of the paper can be reproduced using the following scripts, which run experiments with different training dataset configurations:

```bash
bash run_experiments.sh final lstm 64 1 0.2 2 100 False True False
python scrips/parse_results.py final lstm
```

### Inference
For using a trained model for inference on satellite images, first download the satellite images of the region using the regional exporter from CropHarvest. Then run the following command:

```bash
python scripts/inference_nigeria.py
```
### Map creation
The final maps and the map figures of the paper are created using the following scripts.
```bash
python scripts/create_map.py
python scripts/create_figure_nigeria_map.py
```


## :pray: Acknowledgements
This work was largely based on the amazing work by [NASA Harvest](https://nasaharvest.org/) and their open-source software. In particular, we relied on [togo-crop-mask](https://github.com/nasaharvest/togo-crop-mask) as an initial template and on the [CropHarvest](https://github.com/nasaharvest/cropharvest) package for our final code.

## :bookmark_tabs: Citation

If you find this code, data or our paper useful, please cite us:
    
    ```
