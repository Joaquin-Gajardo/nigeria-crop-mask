import pandas as pd
import geopandas
from tqdm import tqdm
import pathlib as Path
from datetime import date

from .base import BaseSentinelExporter
from src.exporters import GeoWikiExporter
from .utils import EEBoundingBox, bounding_box_from_centre

from typing import Optional, List


class NigeriaSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_geowiki"

    def load_labels(self) -> pd.DataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        geowiki = self.data_folder / "processed" / GeoWikiExporter.dataset / "data.nc"
        assert geowiki.exists(), "GeoWiki processor must be run to load labels"
        return xr.open_dataset(geowiki).to_dataframe().dropna().reset_index()

    def labels_to_bounding_boxes(
        self, num_labelled_points: Optional[int], surrounding_metres: int
    ) -> List[EEBoundingBox]:

        output: List[EEBoundingBox] = []

        for idx, row in tqdm(self.labels.iterrows()):
            output.append(
                bounding_box_from_centre(
                    mid_lat=row["lat"],
                    mid_lon=row["lon"],
                    surrounding_metres=surrounding_metres,
                )
            )

            if num_labelled_points is not None:
                if len(output) >= num_labelled_points:
                    return output
        return output

    def export_for_labels(
        self,
        days_per_timestep: int = 30,
        start_date: date = date(2017, 3, 28),
        end_date: date = date(2018, 3, 28),
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
    ) -> None:
        r"""
        Run the GeoWiki exporter. For each label, the exporter will export

        int( (end_date - start_date).days / days_per_timestep) timesteps of data,

        where each timestep consists of a mosaic of all available images within the
        days_per_timestep of that timestep.

        :param days_per_timestep: The number of days of data to use for each mosaiced image.
        :param start_date: The start data of the data export
        :param end_date: The end date of the data export
        :param num_labelled_points: (Optional) The number of labelled points to export.
        :param surrounding_metres: The number of metres surrounding each labelled point to export
        :param checkpoint: Whether or not to check in self.data_folder to see if the file has
            already been exported. If it has, skip it
        :param monitor: Whether to monitor each task until it has been run
        """
        assert (
            start_date >= self.min_date
        ), f"Sentinel data does not exist before {self.min_date}"

        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            num_labelled_points=num_labelled_points,
            surrounding_metres=surrounding_metres,
        )

        # check for idxs already downloaded
        from pathlib import Path
        existing_idxs = []
        for filepath in Path.iterdir(self.output_folder):
            filename = filepath.as_posix().split('/')[-1]
            if filename.endswith('.tif'):
                existing_idx = int(filename.split('_')[0])
                existing_idxs.append(existing_idx)
        existing_idxs.sort()

        tasks = 0
        last_idx = -1
        for idx, bounding_box in enumerate(bounding_boxes_to_download):
            if (idx not in existing_idxs) and idx > last_idx:
                self._export_for_polygon(
                    polygon=bounding_box.to_ee_polygon(),
                    polygon_identifier=idx,
                    start_date=start_date,
                    end_date=end_date,
                    days_per_timestep=days_per_timestep,
                    checkpoint=checkpoint,
                    monitor=monitor,
                )
                
                tasks += 1
            #else:
            #    print('Already stored locally. Skipping')
            if tasks >= 3000:
                print('Too many tasks for Google Earth Engine. Stopping.')
                break
        print(tasks)