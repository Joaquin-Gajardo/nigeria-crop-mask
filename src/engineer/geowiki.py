from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import xarray as xr

from typing import Optional, List, Union


from src.exporters import GeoWikiExporter, GeoWikiSentinelExporter
from .base import BaseEngineer, BaseDataInstance


@dataclass
class GeoWikiDataInstance(BaseDataInstance):

    crop_probability: float
    neighbouring_array: np.ndarray


class GeoWikiEngineer(BaseEngineer):

    sentinel_dataset = GeoWikiSentinelExporter.dataset
    dataset = GeoWikiExporter.dataset

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        geowiki = data_folder / "processed" / GeoWikiExporter.dataset / "data.nc"
        assert geowiki.exists(), "GeoWiki processor must be run to load labels"
        return xr.open_dataset(geowiki).to_dataframe().dropna().reset_index()

    def get_geospatial_files_per_country(self, data_folder: Path, country: str) -> List[Path]:
        sentinel_files = data_folder / "raw" / self.sentinel_dataset
        all_files = list(sentinel_files.glob("*.tif"))
        
        # Get only those inside country
        df = pd.read_csv(self.savedir / 'identifiers_plus_cropland_prob.csv') # NOTE: see notebook 01 to see how this csv file is created
        geowiki_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs=4326)

        world_map = gpd.read_file('../assets/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
        country_shp = world_map[world_map['SOVEREIGNT'] == country].reset_index(drop=True)
        geowiki_subset_mask = geowiki_points.within(country_shp.loc[0, 'geometry'])
        geowiki_subset = geowiki_points.loc[geowiki_subset_mask]
        
        #Plot
        # fig, ax = plt.subplots(1,1, figsize=(10,10))
        # country_shp.plot(ax=ax)
        # geowiki_subset.plot(ax=ax,color='red')

        ids = geowiki_subset['identifier'].values.tolist()
        files = []
        for file_path in all_files:
            file_info = self.process_filename(file_path.name, True)
            identifier, _ , _ = file_info
            if int(identifier) in ids:
                files.append(file_path)
        
        return files

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
    ) -> Optional[GeoWikiDataInstance]:
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

        # we turn the percentage into a fraction
        crop_probability = overlap.iloc[0].mean_sumcrop / 100

        closest_lon = self.find_nearest(da.x, label_lon)
        closest_lat = self.find_nearest(da.y, label_lat)

        labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values

        # we randomly select another instance
        neighbour_lat, neighbour_lon = self.randomly_select_latlon(
            lat=da.y, lon=da.x, label_lat=label_lat, label_lon=label_lon
        )

        neighbour_np = da.sel(x=neighbour_lon).sel(y=neighbour_lat).values

        if add_ndvi:
            labelled_np = self.calculate_ndvi(labelled_np)
            neighbour_np = self.calculate_ndvi(neighbour_np)

        labelled_array = self.maxed_nan_to_num(
            labelled_np, nan=nan_fill, max_ratio=max_nan_ratio
        )
        # we don't check for the neighbouring array, to prevent unnecessarily removing
        # instances
        neighbouring_array = self.maxed_nan_to_num(neighbour_np, nan=nan_fill)

        if (not is_test) and calculate_normalizing_dict:
            # we won't use the neighbouring array for now, since tile2vec is
            # not really working
            self.update_normalizing_values(labelled_array)
            self.update_normalizing_values_per_class(labelled_array, is_crop=crop_probability>=0.5)

        if labelled_array is not None:
            return GeoWikiDataInstance(
                label_lat=label_lat,
                label_lon=label_lon,
                crop_probability=crop_probability,
                instance_lat=closest_lat,
                instance_lon=closest_lon,
                labelled_array=labelled_array,
                neighbouring_array=neighbouring_array,
            )
        else:
            print("Skipping! Too many nan values")
            return None
