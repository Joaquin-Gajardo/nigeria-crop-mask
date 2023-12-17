# Nigeria Crop Mask 2020

A pixel-wise land type classifier, used to generate a crop mask for Togo
This repository contains code and data to generate a crop mask for Togo.
It was used to deliver a high-resolution (10m) cropland mask in 10 days to help the government distribute aid to smallholder farmers during the COVID-19 pandemic.

## Results

<p align="center">
    <img src="assets/nigeria_cropland_probability_map_image.png" alt="Nigeria map" height="600px"/>
</p>

## Data
It combines a hand-labelled dataset of crop / non-crop images with a [global database of crowdsourced cropland data](https://doi.pangaea.de/10.1594/PANGAEA.873912)
to train a multi-headed LSTM-based model to predict the presence of cropland in a pixel.

The map can be found on [Google Earth Engine](https://code.earthengine.google.com/5d8ff282e63c26610b7cd3b4a989929c).

<p align="center">
    <img src="assets/nigeria_dataset_splits_new.png" alt="Nigeria map" height="600px"/>
</p>


## Pipeline

The main entrypoints into the pipeline are the [scripts](scripts). Specifically:

* [scripts/export.py](scripts/export.py) exports data (locally, or to Google Drive - see below)
* [scripts/process.py](scripts/process.py) processes the raw data
* [scripts/engineer.py](scripts/engineer.py) combines the earth observation data with the labels to create (x, y) training data
* [scripts/models.py](scripts/models.py) trains the models
* [scripts/predict.py](scripts/predict.py) takes a trained model and runs it on exported tif files (the path to these files is defined in the script)

The [split_tiff.py](scripts/split_tiff.py) script is useful to break large exports from Google Earth Engine, which may
be too large to fit into memory.

Once the pipeline has been run, the directory structure of the [data](data) folder should look like the following. If you get errors, a good first check would be to see if any files are missing.

```
data
│   README.md
│
└───raw // raw exports
│   └───togo  // this is included in this repo
│   └───geowiki_landcover_2017  // exported by scripts.export.export_geowiki()
│   └───earth_engine_togo  // exported to Google Drive by scripts.export.export_togo(), and must be copied here
│   │                      // scripts.export.export_togo() expects processed/togo{_evaluation} to exist
│   └───earth_engine_togo_evaluation  // exported to Google Drive by scripts.export.export_togo(), and must be copied here
│   │                                 // scripts.export.export_togo() expects processed/togo{_evaluation} to exist
│   └───earth_engine_geowiki  // exported to Google Drive by scripts.export.export_geowiki_sentinel_ee(), and must be copied here
│                             // scripts.export.export_geowiki_sentinel_ee() expects processed/geowiki_landcover_2017 to exist
│
└──processed  // raw data processed for clarity
│   └───geowiki_landcover_2017 // created by scripts.process.process_geowiki()
│   │                          // which expects raw/geowiki_landcover_2017 to exist
│   └───togo  // created by scripts.process.process_togo()
│   └───togo_evaluation  // created by scripts.process.process_togo()
│
└──features  // the arrays which will be ingested by the model
│   └───geowiki_landcover_2017 // created by scripts.engineer.engineer_geowiki()
│   └───togo  // created by scripts.engineer.engineer_togo()
│   └───togo_evaluation  // created by scripts.engineer.engineer_togo()
│
└──lightning_logs // created by pytorch_lightning when training models
```

### Steps
```
1. cd scripts
2. python -c "from export.py import export_geowiki; export_geowiki()"
3. python -c "from process.py import process_geowiki; process_geowiki()"
4. python -c "from export.py import export_geowiki_sentinel_ee; export_geowiki_sentinel_ee() ## --> this might take up to 10 days and must be done by batches of 3000 tasks at a time (i.e. GEE task limit)."
5. python -c "from process.py import process_togo; process_togo()"
6. python -c "from export.py import export_togo; export_togo()"
7. Repeat 6., but manually changing evaluation_set to False in exporter.export_for_labels inside export_togo() 
8. python -c "from export.py import export_region; export_region()"
9. python -c "from engineer.py import engineer_geowiki; engineer_geowiki()"

10. python -c "from process.py import process_nigeria; process_nigeria()"
11. python -c "from export.py import export_nigeria; export_nigeria()"
11. python -c "from export.py import export_gdrive_nigeria; export_gdrive_nigeria()"
12. python -c "from engineer_nigeria.py import engineer_nigeria; engineer_nigeria()"
```

## Setup

### Enviroment
[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `nigeria-crop-mask` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate nigeria-crop-mask
```
To use a GPU enviroment, run:
```bash
conda env create -f environment-gpu.yml
conda activate nigeria-crop-mask-gpu
```

#### Manual installation
For GPU, my gpu [can't use](https://discuss.pytorch.org/t/torch-being-installed-with-cpu-only-even-when-i-have-a-gpu/135060/8) cuda toolkit 10.2, and otherwise always installs pytorch with cpu. The following works, based on [pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows):
```
conda install python=3.7
#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch -c conda-forge
#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge pytorch-lightning"<=1.0" # 0.8.5 works with one gpu, put not with ddp
#conda install -c conda-forge pytorch-lightning=1.8 # breaks due to need of doing self.save_hyperparameters() instead of self.hparams = hparams
conda install matplotlib geopandas xarray tqdm scikit-learn rasterio jupyter cartopy
conda install -c conda-forge earthengine-api wandb
conda install google-auth-oauthlib
```
However, GPU doesn't make it faster but slower actually, compared to just using more cores (n_workers) for the data loaders (2x speedup).

Install geemap environment:
```bash
conda create -n geemap
conda activate geemap
conda install geopandas
conda install geemp -c conda-forge
```

### Earth Engine

Earth engine is used to export data. To use it, once the conda environment has been activated, run

```bash
earthengine authenticate
```

and follow the instructions. To test that everything has worked, run

```bash
python -c "import ee; ee.Initialize()"
```

Note that Earth Engine exports files to Google Drive by default (to the same google account used sign up to Earth Engine).

Running exports can be viewed (and individually cancelled) in the `Tabs` bar on the [Earth Engine Code Editor](https://code.earthengine.google.com/).
For additional support the [Google Earth Engine forum](https://groups.google.com/forum/#!forum/google-earth-engine-developers) is super
helpful.

Exports from Google Drive should be saved in [`data/raw`](data/raw).
This happens by default if the [GDrive](src/exporters/gdrive.py) exporter is used.

### Inference
For using a trained model for inference on satellite images, first download the satellite images of the region using the regional exporter from CropHarvest. Then run the following command:

```bash
python scripts/inference_nigeria.py
```

## Acknowledgements
This work was largely based on the amazing work by [NASA Harvest](https://nasaharvest.org/) work. In particular, we used their [togo-crop-mask](https://github.com/nasaharvest/togo-crop-mask) as a template and relied heavily on the the [CropHarvest](https://github.com/nasaharvest/cropharvest) package.

## Citation

If you find this code useful, please cite the following paper:

The hand-labeled training and test data used in the above paper can be found at: https://doi.org/10.5281/zenodo.3836629
