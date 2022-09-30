import numpy as np
import geopandas as gdp
from pathlib import Path

from .base import BaseProcessor

class NigeriaProcessor(BaseProcessor):

    dataset = "nigeria_farmlands"

    def __init__(self, data_folder: Path) -> None:
        super().__init__(data_folder)

    @staticmethod
    def process_shapefile(filepath: Path) -> gdp.GeoDataFrame:
        df = gdp.read_file(filepath)

        # Not consider those unsure (-1)
        df = df[(df.review_joa == 1.0) | (df.review_joa == 0.0)]
        df["is_crop"] = np.where(df['review_joa'] == 1, 1, 0) # extra check
        df["lon"] = df.geometry.centroid.x
        df["lat"] = df.geometry.centroid.y
        df.reset_index(drop=True, inplace=True)

        return df[["is_crop", "geometry", "lat", "lon"]]

    def process(self) -> None:

        shapefile = self.raw_folder / 'nigeria_dataset_v2_joaquin.shp'
        
        df = self.process_shapefile(shapefile)
        df["index"] = df.index
        
        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")

