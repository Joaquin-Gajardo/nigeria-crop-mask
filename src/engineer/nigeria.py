from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import geopandas
from datetime import datetime

from typing import Optional


from src.processors import NigeriaProcessor
from src.exporters import NigeriaSentinelExporter
from .base import BaseEngineer, BaseDataInstance


@dataclass
class NigeriaDataInstance(BaseDataInstance):
    is_crop: int


class NigeriaEngineer(BaseEngineer):

    sentinel_dataset = NigeriaSentinelExporter.dataset
    dataset = NigeriaProcessor.dataset

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        nigeria = data_folder / "processed" / NigeriaProcessor.dataset / "data.geojson"
        assert nigeria.exists(), "Nigeria processor must be run to load labels"
        return geopandas.read_file(nigeria)

    def process_single_file(
        self,
        path_to_file: Path,
        nan_fill: float,
        max_nan_ratio: float,
        add_ndvi: bool,
        calculate_normalizing_dict: bool,
        start_date: datetime,
        days_per_timestep: int,
        is_test: bool,
    ) -> Optional[NigeriaDataInstance]:
        r"""
        Return a tuple of np.ndarrays of shape [n_timesteps, n_features] for
        1) the anchor (labelled)
        """

        da = self.load_tif(
            path_to_file, days_per_timestep=days_per_timestep, start_date=start_date
        )

        # first, we find the label encompassed within the da

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        overlap = self.labels[
            (
                (self.labels.lon <= max_lon)
                & (self.labels.lon >= min_lon)
                & (self.labels.lat <= max_lat)
                & (self.labels.lat >= min_lat)
            )
        ]
        if len(overlap) == 0:
            return None

        label_lat = overlap.iloc[0].lat
        label_lon = overlap.iloc[0].lon

        closest_lon = self.find_nearest(da.x, label_lon)
        closest_lat = self.find_nearest(da.y, label_lat)

        labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values

        if add_ndvi:
            labelled_np = self.calculate_ndvi(labelled_np)

        labelled_array = self.maxed_nan_to_num(
            labelled_np, nan=nan_fill, max_ratio=max_nan_ratio
        )

        if (not is_test) and calculate_normalizing_dict:
            self.update_normalizing_values(labelled_array)
            self.update_normalizing_values_per_class(labelled_array, is_crop=int(overlap.iloc[0].is_crop))

        if labelled_array is not None:
            return NigeriaDataInstance(
                label_lat=label_lat,
                label_lon=label_lon,
                instance_lat=closest_lat,
                instance_lon=closest_lon,
                labelled_array=labelled_array,
                is_crop=int(overlap.iloc[0].is_crop),
            )
        else:
            print("Skipping! Too many nan values")
            return None
