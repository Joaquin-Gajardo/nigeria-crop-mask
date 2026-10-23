# The data folder

The pipeline reads from and writes to this directory (by default). The main contents are:

- `features/nigeria-cropharvest/`: the Nigeria dataset. The labels and data splits are included in the repository, while the data arrays (`features/arrays/*.h5` and `features/normalizing_dict.h5`) need to be downloaded from Google Drive (see the main README).
- `cropharvest/`: the CropHarvest dataset, from which the Geowiki data is used. The features need to be downloaded before training (see the main README), while the labels are downloaded automatically.
- `lightning_logs/`: model checkpoints and logs written during training. The checkpoint of the model used for the paper maps can be downloaded from Google Drive and placed here.
- `predictions/`: the predicted maps written by `scripts/inference_nigeria.py` and `scripts/create_map_nigeria.py`.
