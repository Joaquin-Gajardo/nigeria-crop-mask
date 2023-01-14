import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
from itertools import permutations

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.exporters import GeoWikiExporter
from src.exporters.sentinel.cloudfree import BANDS
from src.processors import TogoProcessor, NigeriaProcessor, NigeriaProcessorNew
from src.utils import STR2BB
from src.engineer.geowiki import GeoWikiDataInstance, GeoWikiEngineer
from src.engineer.togo import TogoDataInstance
from src.engineer.nigeria import NigeriaDataInstance

from typing import cast, Tuple, Optional, List, Dict, Sequence, Union, Type, TypeVar


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
        data_folder: Path,
        subset: str,
        crop_probability_threshold: Optional[float],
        include_geowiki: bool,
        include_togo: bool,
        include_nigeria: bool,
        remove_b1_b10: bool,
        normalizing_dict: Optional[Dict] = None,
    ) -> None:

        self.normalizing_dict: Optional[Dict] = None
        self.data_folder = data_folder
        self.features_dir = data_folder / "features"
        self.bands_to_remove = ["B1", "B10"]
        self.remove_b1_b10 = remove_b1_b10

        assert subset in ["training", "validation", "testing"]
        self.subset_name = subset

        self.crop_probability_threshold = crop_probability_threshold

        # if (subset == "testing") and (self.features_dir / NigeriaProcessor.dataset / subset).exists(): # If the test set for Nigeria exist it will use it instead of the 
        #     print("Evaluating using the Nigeria evaluation dataset!")

        #     assert normalizing_dict is not None # we want to normalize our test set with the training and val data statistics 
        #     self.normalizing_dict = normalizing_dict

        #     self.pickle_files, _ = self.load_files_and_normalizing_dict(
        #         self.features_dir / NigeriaProcessor.dataset, subset
        #     )
        #     print(f'Number of instances in {NigeriaProcessor.dataset} test set: {len(self.pickle_files)}')
        #     print(self.normalizing_dict)

        # else:
        assert (
            max(include_geowiki, include_togo, include_nigeria) is True
        ), "At least one dataset must be included"

        files_and_dicts: List[Tuple[List[Path], Optional[Dict]]] = []

        if include_geowiki:

            geowiki_files, geowiki_nd = self.load_files_and_normalizing_dict(
                self.features_dir / GeoWikiExporter.dataset , self.subset_name
            )
            print(f'Number of geowiki instances in {self.subset_name} set: {len(geowiki_files)}')

            files_and_dicts.append((geowiki_files, geowiki_nd))

        if include_togo:
            togo_files, togo_nd = self.load_files_and_normalizing_dict(
                self.features_dir / TogoProcessor.dataset, self.subset_name,
            )
            files_and_dicts.append((togo_files, togo_nd))
            
        if include_nigeria:
            nigeria_files, nigeria_nd = self.load_files_and_normalizing_dict(
                self.features_dir / NigeriaProcessorNew.dataset, self.subset_name,
            )
            files_and_dicts.append((nigeria_files, nigeria_nd))

        if normalizing_dict is None:
            # if a normalizing dict wasn't passed to the constructor,
            # then we want to make our own
            self.normalizing_dict = self.adjust_normalizing_dict(
                [(len(x[0]), x[1]) for x in files_and_dicts]
            )
        else:
            self.normalizing_dict = normalizing_dict

        pickle_files: List[Path] = []
        for files, _ in files_and_dicts:
            pickle_files.extend(files)
        self.pickle_files = pickle_files
        print(f"{len(self.pickle_files)} files in {subset}")

    @property
    def num_output_classes(self) -> int:
        return 1

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the anchor, neighbour, distant tensors
        """
        target_file = self.pickle_files[index]

        # first, we load up the target file
        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        if isinstance(target_datainstance, GeoWikiDataInstance):
            if self.crop_probability_threshold is None:
                label = target_datainstance.crop_probability
            else:
                label = int(
                    target_datainstance.crop_probability
                    >= self.crop_probability_threshold
                )
        elif isinstance(target_datainstance, TogoDataInstance):
            label = target_datainstance.is_crop
        elif isinstance(target_datainstance, NigeriaDataInstance):
            label = target_datainstance.is_crop
        else:
            raise RuntimeError(
                f"Unrecognized data instance type {type(target_datainstance)}"
            )

        weight = 0
        if target_datainstance.isin(STR2BB["Nigeria"]):
            weight = 1

        return (
            torch.from_numpy(
                self.remove_bands(x=self._normalize(target_datainstance.labelled_array))
            ).float(),
            torch.tensor(label).float(),
            torch.tensor(weight).long(),
        )

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

    @property
    def num_input_features(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[1]

    @property
    def num_timesteps(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[0]

    def remove_bands(self, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands]
        """
        if self.remove_b1_b10:
            indices_to_remove: List[int] = []
            for band in self.bands_to_remove:
                indices_to_remove.append(BANDS.index(band))

            bands_index = 1 if len(x.shape) == 2 else 2
            indices_to_keep = [
                i for i in range(x.shape[bands_index]) if i not in indices_to_remove
            ]
            if len(x.shape) == 2:
                # timesteps, bands
                return x[:, indices_to_keep]
            else:
                # batches, timesteps, bands
                return x[:, :, indices_to_keep]
        else:
            return x

    @staticmethod
    def load_files_and_normalizing_dict(
        features_dir: Path, subset_name: str
    ) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]]]:
        pickle_files = list((features_dir / subset_name).glob("*.pkl"))

        # try loading the normalizing dict. By default, if it exists we will use it
        if (features_dir / "normalizing_dict.pkl").exists():
            with (features_dir / "normalizing_dict.pkl").open("rb") as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None

        return pickle_files, normalizing_dict

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.normalizing_dict["mean"]) / self.normalizing_dict[
                "std"
            ]

    def __len__(self) -> int:
        return len(self.pickle_files)


GeowikiDatasetType = TypeVar('GeowikiDatasetType', bound='Parent') # for typing

class GeowikiDataset(Dataset):
    
    dataset_name: str = GeoWikiEngineer.dataset
    csv_file: str = 'geowiki_labels_country_crs4326.csv'

    def __init__(self, data_folder: Path = Path('../data'),
                countries_subset: Optional[List[str]] = None,
                countries_to_weight: Optional[List[str]] = None,
                crop_probability_threshold: float = 0.5,
                remove_b1_b10: bool = True,
                normalizing_dict: Optional[Dict] = None,
                labels: Optional[pd.DataFrame] = None # if this is passed csv_file will be ignored
                ) -> None:
        
        # Attributes
        self.data_folder = data_folder
        self.dataset_dir = data_folder / "features" / self.dataset_name
        self.countries_subset = countries_subset
        self.countries_to_weight = countries_to_weight
        self.bands_to_remove = ["B1", "B10"]
        self.remove_b1_b10 = remove_b1_b10
        self.crop_probability_threshold = crop_probability_threshold
        
        # Functions
        if labels is None:
            self.labels = pd.read_csv(self.dataset_dir / self.csv_file)
            self.labels.loc[self.labels['country'].isnull(), 'country'] = 'unknown'
            if self.countries_subset:
                self.labels = self.labels[self.labels['country'].str.lower().isin(list(map(str.lower, self.countries_subset)))].reset_index(drop=True)
        else:
            self.labels = labels
        self.pickle_files = self.get_pickle_files_paths(self.dataset_dir / 'all')
        self.file_identifiers_countries_to_weight = self.get_file_ids_for_countries(self.countries_to_weight)
        print('length labels:', len(self.labels))
        print('length pickle files:', len(self.pickle_files))
        print('length local ids:', len(self.file_identifiers_countries_to_weight))

        # Normalizing dictionary
        if normalizing_dict is None:
            self.normalizing_dict = self.get_normalizing_dict()
        else:
            self.normalizing_dict = normalizing_dict    
        print(self.normalizing_dict)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the data, label, and weight tensors.
        """
        target_file = self.pickle_files[index]
        identifier = int(target_file.name.split('_')[0])

        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        if isinstance(target_datainstance, GeoWikiDataInstance):
            if self.crop_probability_threshold is None:
                label = target_datainstance.crop_probability
            else:
                label = int(target_datainstance.crop_probability >= self.crop_probability_threshold)
        else:
            raise RuntimeError(f"Unrecognized data instance type {type(target_datainstance)}")

        is_local = 0
        if identifier in self.file_identifiers_countries_to_weight:
            is_local = 1

        return (
            torch.from_numpy(
                self.remove_bands(x=self._normalize(target_datainstance.labelled_array))
            ).float(),
            torch.tensor(label).float(),
            torch.tensor(is_local).long(),
        )

    @property
    def num_output_classes(self) -> int:
        return 1

    @property
    def num_input_features(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[1]

    @property
    def num_timesteps(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[0]

    def remove_bands(self, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands]
        """
        if self.remove_b1_b10:
            indices_to_remove: List[int] = []
            for band in self.bands_to_remove:
                indices_to_remove.append(BANDS.index(band))

            bands_index = 1 if len(x.shape) == 2 else 2
            indices_to_keep = [i for i in range(x.shape[bands_index]) if i not in indices_to_remove]
            if len(x.shape) == 2:
                # timesteps, bands
                return x[:, indices_to_keep]
            else:
                # batches, timesteps, bands
                return x[:, :, indices_to_keep]
        else:
            return x

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]

    @staticmethod
    def load_files_and_normalizing_dict(
        features_dir: Path, subset_name: str='training', file_name: str="normalizing_dict.pkl"
    ) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]]]:
        pickle_files = list((features_dir / subset_name).glob("*.pkl"))

        # try loading the normalizing dict. By default, if it exists we will use it
        if (features_dir / file_name).exists():
            with (features_dir / file_name).open("rb") as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None

        return pickle_files, normalizing_dict

    def search_normalizing_dict(self, default_file_name: str="normalizing_dict.pkl") -> Optional[Path]:
        '''
        Searches for the normalizing dict file in the self.data_dir directory and returns its path. Returns None if it was not found.
        '''
        prefix = default_file_name.split('.')[0]
        
        if not self.countries_subset:
            file_path = self.dataset_dir / default_file_name
            if file_path.exists():
                print(f'Found normalizing dict {file_path.name}')
                return file_path
        elif len(self.countries_subset) == 1 and self.countries[0].lower() == 'africa':
            raise NotImplementedError # TODO

        else:
            assert len(self.countries_subset) < 10, 'Execution time will be too big!' # TODO: add warning when passing subset to constructor
            countries_permutations = list(permutations(self.countries_subset))
            countries_permutations = ['_'.join(permutation) for permutation in countries_permutations]
            for permutation in countries_permutations:
                file_name = f"{prefix}_{permutation}.pkl"
                file_path = self.dataset_dir / file_name
                if file_path.exists():
                    print(f'Found normalizing dict {file_name}')
                    return file_path
        print('Normalizing dict not found.')
        return None

    def get_normalizing_dict(self, save: bool=False) -> Dict:
        # Return dict if it was found or create it and save
        default_file_name = "normalizing_dict.pkl"
        file_path = self.search_normalizing_dict(default_file_name)
        if file_path:
            print('Loading normalizing dict.')
            return self.load_files_and_normalizing_dict(self.dataset_dir, file_name=file_path.name)[1]
        else:
            print('Calculating normalizing dict...')
            assert len(self) == len(self.pickle_files), 'Length of self.labels must be the same as of the list of pickle files.'
            geowiki_engineer = GeoWikiEngineer(Path('../data'))
            
            for file_path in tqdm(self.pickle_files):
                identifier = int(file_path.name.split('_')[0])
                with file_path.open("rb") as f:   
                    target_datainstance = pickle.load(f)
                geowiki_engineer.update_normalizing_values(target_datainstance.labelled_array)

            normalizing_dict = geowiki_engineer.calculate_normalizing_dict()

            # Write file
            if save:    
                if self.countries_subset:
                    prefix = default_file_name.split('.')[0]
                    countries_str = '_'.join(self.countries_subset)
                    file_name = f"{prefix}_{countries_str}.pkl"
                else:
                    file_name = default_file_name
                file_path = self.data_dir / file_name
                print('Saving normalizing dict', file_path.name)
                with file_path.open("wb") as f:
                    pickle.dump(normalizing_dict, f)

            return normalizing_dict
            
    def get_pickle_files_paths(self, folder_path: Path) -> Tuple[List[Path]]:
        file_paths = self.labels.filename.tolist()
        print('Checking for data files')
        pickle_files = [path for path in tqdm(folder_path.glob('*.pkl')) if path.name in file_paths]
        self._check_label_files(pickle_files)
        return pickle_files

    def _check_label_files(self, pickle_files) -> None:
        same_files = set([file.name for file in pickle_files]) == set(self.labels.filename.tolist())
        assert same_files, "Some pickle files of the labels were not found!"
        print('All pickle files were found!')

    def get_file_ids_for_countries(self, countries_list: List[str]) -> List[int]:
        file_ids = []
        if countries_list:
            countries_list_lowercase = list(map(str.lower, countries_list))
            file_ids.extend(self.labels[self.labels['country'].str.lower().isin(countries_list_lowercase)]['identifier'].tolist())
        return file_ids

    @classmethod
    def train_val_split(cls: Type[GeowikiDatasetType], class_instance: Type[GeowikiDatasetType],
                        train_size: float=0.8, stratify_column: Optional[str]=None
                        ) -> Tuple[GeowikiDatasetType]:
        # Made it a class method to be able to generate two child instances of the class.
        # Haven't figured out a better way for now than passing the parent instance as an argument.

        # Split labels dataframe
        stratify = None if not stratify_column else class_instance.labels[stratify_column]
        df_train, df_val = train_test_split(class_instance.labels, train_size=train_size, stratify=stratify, random_state=42)
        df_train.reset_index(drop=True, inplace=True) 
        df_val.reset_index(drop=True, inplace=True)

        # Create two new GeowikiDataset instances (train and val)
        print('Train split')
        train_dataset = cls(countries_to_weight=class_instance.countries_to_weight,
                        normalizing_dict=class_instance.normalizing_dict, labels=df_train)
        print('Val split')
        val_dataset = cls(countries_to_weight=class_instance.countries_to_weight,
                        normalizing_dict=class_instance.normalizing_dict, labels=df_val)
        return train_dataset, val_dataset

    def get_file_by_identifier(self):
        raise NotImplementedError
