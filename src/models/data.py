import h5py
from itertools import permutations
from pathlib import Path
from typing import cast, Optional, List, Tuple, Dict, Type, TypeVar, Sequence
from tqdm import tqdm
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from cropharvest.datasets import CropHarvestLabels
from cropharvest.columns import RequiredColumns
from cropharvest.config import FEATURES_DIR
from cropharvest.engineer import Engineer
from cropharvest.utils import load_normalizing_dict
from cropharvest.bands import BANDS


S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','NDVI']

here = Path(__file__).parent
ROOT = here.parent.parent

class BaseCropHarvestDataset:
    root: Path
    filepaths: List
    y_vals: List

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int):
        file = h5py.File(self.filepaths[index], "r")
        return self._normalize(file.get("array")[:]), self.y_vals[index]
    
    def as_array(self, flatten_x=False, S2_features_only=False):
        indices_to_sample = list(range(len(self)))
        X, Y = zip(*[self[i] for i in indices_to_sample])
        X_np, y_np = np.stack(X), np.stack(Y)
        if S2_features_only:
            indices = [BANDS.index(band) for band in S2_BANDS]
            X_np = np.take(X_np, indices, axis=2)
        if flatten_x:
            X_np = self._flatten_array(X_np)
        return X_np, y_np

    def _path_from_row(self, row):
        path = self.root / f"features/arrays/{row[RequiredColumns.INDEX]}_{row[RequiredColumns.DATASET]}.h5"
        if not path.exists():
            return None
        return path

    def _discard_missing_files(self):
        self.labels['path'] = self.labels.apply(lambda row: self._path_from_row(row), axis=1)
        self.labels = self.labels[~self.labels['path'].isna()].reset_index(drop=True)
        
    def _normalize(self, array):
        if not self.normalizing_dict:
            return array
        return (array - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]
            
    @staticmethod
    def _flatten_array(array):
        return array.reshape(array.shape[0], -1)


class NigeriaCropHarvestDataset(BaseCropHarvestDataset):
    # TODO: Implement dataset creation in src code. This was created with CropHarvest.Engineer in notebook 14. 
    def __init__(self, root='../data/features/nigeria-cropharvest', split='testing'):    
        self.root = Path(root)
        self._labels = CropHarvestLabels(self.root).as_geojson()
        # Slice per subset
        assert split in ["training", "validation", "testing", "trainval"], \
            'Split must be either "training", "validation" or "testing", or "trainval" (for final inference).'
        if split == 'trainval':
            self.labels = self._labels[(self._labels['new_set'] == 'training') | (self._labels['new_set'] == 'validation')].reset_index(drop=True)
        else:
            self.labels = self._labels[self._labels['new_set'] == split].reset_index(drop=True)
        self._discard_missing_files()  
        self.filepaths = self.labels['path'].tolist()
        self.y_vals = self.labels['is_crop'].tolist()
        # Normalizing dict created on notebook 14
        self.normalizing_dict = load_normalizing_dict(self.root/ FEATURES_DIR / 'normalizing_dict.h5')
    

GeowikiCropHarvestDatasetType = TypeVar('GeowikiCropHarvestDatasetType', bound='Parent') # for typing

class GeowikiCropHarvestDataset(BaseCropHarvestDataset):
    def __init__(self, 
                root: str,
                countries_subset: Optional[List[str]] = None,
                labels: Optional[pd.DataFrame] = None, # if this is passed csv_file will be ignored
                normalizing_dict: Optional[Dict] = None,
                ) -> None:
            
        self.root = Path(root)
        self.countries_subset = countries_subset

        # Labels
        if labels is None:
            cropharvest_labels = CropHarvestLabels(root, download=True)
            cropharvest_df = cropharvest_labels.as_geojson()
            self.labels = cropharvest_df[cropharvest_df['dataset'] == 'geowiki-landcover-2017'].reset_index(drop=True)
            self.labels = self._add_country()
            self._discard_missing_files() # only around 24761/35866 geowiki files come with cropharvest
            # Country subset
            if self.countries_subset:
                self.labels = self.labels[self.labels.country.str.lower().isin(list(map(str.lower, self.countries_subset)))]
        else: # Comes from split
            self.labels = labels

        self.filepaths = self.labels['path'].tolist()
        self.y_vals = self.labels['is_crop'].tolist()
        if normalizing_dict is None:
            self.normalizing_dict = self.get_normalizing_dict()
        else:
            self.normalizing_dict = normalizing_dict  

    def _add_country(self):
        world_map = gpd.read_file(ROOT / 'assets' / 'ne_50m_admin_0_countries' / 'ne_50m_admin_0_countries.shp') # TODO: maybe move to utils to access from different places with a fixed path to assets
        labels = self.labels.copy()
        join_df = gpd.sjoin(labels, world_map, how='left', op="within")[['index', 'ADMIN']] # spatial merge and keep only necessary columns
        new_labels = pd.merge(labels, join_df, on='index', how='left')
        new_labels.rename(columns={'ADMIN': 'country'}, inplace=True)
        new_labels.loc[new_labels.country.isnull(), 'country'] = 'unknown' # convert NaN country entries to strings
        return new_labels
        
    def _search_normalizing_dict(self, default_file_name: str="geowiki_normalizing_dict.h5") -> Optional[Path]:
        '''Searches for the normalizing dict file in the self.data_dir directory and returns its path. Returns None if it was not found.'''
        prefix = default_file_name.split('.')[0]
        file_path = self.root / FEATURES_DIR / default_file_name
        if not self.countries_subset:
            if file_path.exists():
                print(f'Found normalizing dict {file_path.name}')
                return file_path
        elif len(self.countries_subset) == 1 and self.countries_subset[0].lower() == 'africa':
            raise NotImplementedError # TODO
        else:
            assert len(self.countries_subset) < 10, 'Execution time will be too big!' # TODO: add warning when passing subset to constructor
            countries_permutations = list(permutations(self.countries_subset))
            countries_permutations = ['_'.join(permutation) for permutation in countries_permutations]
            for permutation in countries_permutations:
                file_name = f"{prefix}_{permutation}.h5"
                file_path = file_path.parent / file_name
                if file_path.exists():
                    print(f'Found normalizing dict {file_name}')
                    return file_path
        print('Normalizing dict not found.')
        return None

    def get_normalizing_dict(self, save: bool=True):
        nd_name = "geowiki_normalizing_dict.h5"
        nd_path = self._search_normalizing_dict(nd_name)
        if nd_path:
            print(f'Loading normalizing dict {nd_path.name}')
            normalizing_dict = load_normalizing_dict(nd_path)
        else:
            print('Calculating normalizing dict')
            geowiki_engineer = Engineer(self.root)
            for file_path in tqdm(self.filepaths):
                with h5py.File(file_path, "r") as file:
                    array = file.get("array")[:]
                geowiki_engineer.update_normalizing_values(array)

            normalizing_dict = geowiki_engineer.calculate_normalizing_dict()

            # Write file
            if save and normalizing_dict is not None:
                if self.countries_subset: # modify nd file name
                    prefix = nd_name.split('.')[0]
                    countries_str = '_'.join(self.countries_subset)
                    nd_name = f"{prefix}_{countries_str}.h5"
                nd_path = self.root / FEATURES_DIR / nd_name
                print('Saving normalizing dict', nd_path.name)
                hf = h5py.File(nd_path, "w")
                for key, val in normalizing_dict.items():
                    hf.create_dataset(key, data=val)
                hf.close()
            
        return normalizing_dict
    
    @classmethod
    def train_val_split(cls: Type[GeowikiCropHarvestDatasetType], class_instance: Type[GeowikiCropHarvestDatasetType],
                        train_size: float=0.8, stratify_column: Optional[str]=None
                        ) -> Tuple[GeowikiCropHarvestDatasetType]:
        # Made it a class method to be able to generate two child instances of the class.
        # Haven't figured out a better way for now than passing the parent instance as an argument.

        # Split labels dataframe
        stratify = None if not stratify_column else class_instance.labels[stratify_column] # Could stratify on label crop or no-crop but it doesn't make a big difference
        df_train, df_val = train_test_split(class_instance.labels, train_size=train_size, stratify=stratify, random_state=42)
        df_train.reset_index(drop=True, inplace=True) 
        df_val.reset_index(drop=True, inplace=True)

        # Create two new GeowikiDataset instances (train and val)
        print('Creating Geowiki train split')
        train_dataset = cls(root=class_instance.root, labels=df_train,
                        normalizing_dict=class_instance.normalizing_dict)
        print('Creating Geowiki val split')
        val_dataset = cls(root=class_instance.root, labels=df_val,
                        normalizing_dict=class_instance.normalizing_dict)
        return train_dataset, val_dataset


class LandTypeClassificationDataset(Dataset):
    r"""
    A dataset for land-type classification data.
    Iterating through this dataset returns a tuple
    (x, y, weight), where weight is an optionally added
    geographically defined weight.

    The dataset should be called through the model - the parameters
    are defined there.
    """

    def __init__(
        self,
        subset: str,
        include_geowiki: bool,
        include_nigeria: bool,
        evaluating: bool = False,
        geowiki_dataset: Optional[GeowikiCropHarvestDataset] = None,
        nigeria_dataset: Optional[GeowikiCropHarvestDataset] = None,
        normalizing_dict: Optional[Dict] = None,
        S2_features_only: bool = False,        
    ) -> None:

        assert subset in ["training", "validation", "testing", "trainval"], \
            'Split must be either "training", "validation" or "testing", or "trainval" (for final inference).'
        self.subset_name = subset
        self.S2_features_only = S2_features_only
        if self.S2_features_only:
            self.band_indices = [BANDS.index(band) for band in S2_BANDS]
                
        self.target_country_borders = gpd.read_file(ROOT / 'assets' / 'nigeria_borders.shp') # Assumes target country is Nigeria. TODO: take from geowiki_set.countries_to_weight and natural earth countries shapefiles

        self.datasets: Dict[str: gpd.GeoDataFrame] = dict()

        # To evaluate/test. We use Nigeria dataset validation files (for dev) or testing folder (for final results)
        if (subset == "testing") or (evaluating and subset == "validation"):
            assert normalizing_dict is not None, "We want to normalize our test set with the training and val data statistics!" 
            self.normalizing_dict = normalizing_dict

            self.filepaths = nigeria_dataset.filepaths
            self.y_vals = nigeria_dataset.y_vals
            self.normalizing_dict = nigeria_dataset.normalizing_dict
            self.datasets['nigeria'] = nigeria_dataset.labels
            print(f'Number of instances in Nigeria {subset} set: {len(self.filepaths)}')

        # For training and validation
        else:
            assert (
                max(include_geowiki, include_nigeria) is True
            ), "At least one dataset must be included"

            files_and_dicts: List[Tuple[List[Path], Optional[Dict]]] = []
    
            ### Grab files and normalizing dicts from each dataset ###
            if include_geowiki:
                geowiki_filepaths = geowiki_dataset.filepaths
                geowiki_y_vals = geowiki_dataset.y_vals
                geowiki_nd = geowiki_dataset.normalizing_dict
                self.datasets['geowiki-landcover-2017'] = geowiki_dataset.labels
                files_and_dicts.append((geowiki_filepaths, geowiki_y_vals, geowiki_nd))
                print(f'Number of instances in Geowiki {subset} set: {len(geowiki_filepaths)}')

            if include_nigeria:
                nigeria_filepaths = nigeria_dataset.filepaths
                nigeria_y_vals = nigeria_dataset.y_vals
                nigeria_nd = nigeria_dataset.normalizing_dict
                self.datasets['nigeria'] = nigeria_dataset.labels
                files_and_dicts.append((nigeria_filepaths, nigeria_y_vals, nigeria_nd))
                print(f'Number of instances in Nigeria {subset} set: {len(nigeria_filepaths)}')

            ### Combine the datasets ###
            if normalizing_dict is None:
                # if a normalizing dict wasn't passed to the constructor,
                # then we want to make our own. E.g. for validation set, don't need to combine it again.
                self.normalizing_dict = self.adjust_normalizing_dict([(len(x[0]), x[2]) for x in files_and_dicts]) # pass on len of each dataset and their normalizing dicts
            else:
                self.normalizing_dict = normalizing_dict

            filepaths: List[Path] = []
            y_vals: List[int] = []
            for files, y_val, nd in files_and_dicts:
                filepaths.extend(files)
                y_vals.extend(y_val)
            self.filepaths = filepaths
            self.y_vals = y_vals
            print(f"Total number of files used for {subset}: {len(self.filepaths)}")

    @property
    def num_output_classes(self) -> int:
        return 1

    @property
    def num_input_features(self) -> int: 
        # assumes the first value in the tuple is x
        assert len(self.filepaths) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[1] #TODO: check how this works if we do flatten the input or not (sol: index via -1)

    @property
    def num_timesteps(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.filepaths) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[0]
    
    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the data, label, and weight tensors.
        """
        # Load up target file
        target_filepath = self.filepaths[index]
        target_file = h5py.File(target_filepath, "r")

        # Infer dataset name from path
        identifier, dataset_name = target_filepath.stem.split('_')

        # Crop probability threshold
        # --> not used, as cropharvest geowiki has hard labels anyways

        # Look up label in self.labels dict
        labels = self.datasets[dataset_name]
        row = labels.loc[labels['index'] == int(identifier)]
        assert len(row) == 1 # should be unique
        lat = row.iloc[0]['lat']
        lon = row.iloc[0]['lon']

        # Weight if point falls in Nigeria (for multiheaded LSTM)
        weight = 0 
        point = Point(lon, lat)
        if self.target_country_borders.contains(point).bool():
            weight = 1

        data = torch.from_numpy(self._normalize(target_file.get("array")[:])).float()
        if self.S2_features_only:
            data = data[:, self.band_indices]
        label = torch.tensor(self.y_vals[index]).float()
        weight = torch.tensor(weight).long()
        
        return data, label, weight

    @staticmethod
    def adjust_normalizing_dict(
        dicts: Sequence[Tuple[int, Optional[Dict[str, np.ndarray]]]]
    ) -> Optional[Dict[str, np.ndarray]]:

        for length, single_dict in dicts:
            if single_dict is None:  # This means if the first normalizing dict (geowiki or Togo if we don't include geowiki) is missing we won't use any (we return None). #NOTE: beware of behaviour if I had more datasets like not geowiki and only use Nigeria or Togo.
                return None

        dicts = cast(Sequence[Tuple[int, Dict[str, np.ndarray]]], dicts)

        new_total = sum([x[0] for x in dicts])

        new_mean = (
            sum([single_dict["mean"] * length for length, single_dict in dicts])
            / new_total
        )

        new_variance = (
            sum(
                [
                    (single_dict["std"] ** 2 + (single_dict["mean"] - new_mean) ** 2)
                    * length
                    for length, single_dict in dicts
                ]
            )
            / new_total
        )

        return {"mean": new_mean, "std": np.sqrt(new_variance)}

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.normalizing_dict["mean"]) / self.normalizing_dict[
                "std"
            ]
        
    def as_array(self, flatten_x=False, S2_features_only=False):
        indices_to_sample = list(range(len(self)))
        X, Y, weights = zip(*[self[i] for i in indices_to_sample])
        X = [x.unsqueeze(0) for x in X]
        Y = [y.item() for y in Y]
        weights = [weight.item() for weight in weights]
        X_np, y_np, weights_np = torch.cat(X).numpy(), torch.tensor(Y).numpy(), np.array(weights)
        if S2_features_only:
            indices = [BANDS.index(band) for band in S2_BANDS]
            X_np = np.take(X_np, indices, axis=2)
        if flatten_x:
            X_np = self._flatten_array(X_np)
        return X_np, y_np
            
    @staticmethod
    def _flatten_array(array):
        return array.reshape(array.shape[0], -1)
