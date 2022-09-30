import pandas as pd
import geopandas
from tqdm import tqdm
from datetime import date, timedelta

from .base import BaseSentinelExporter
from src.processors.nigeria import NigeriaProcessor
from .utils import EEBoundingBox, bounding_box_from_centre

from typing import Optional, List


class NigeriaSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_nigeria"

    # Nigeria farmlands data is from 2018 and verification by photointerpretation was done with Google Satellite (from unknown dates...)
    data_date = date(2018, 3, 28)

    def load_labels(self) -> pd.DataFrame:
        nigeria = self.data_folder / "processed" / NigeriaProcessor.dataset / "data.geojson"
        assert nigeria.exists(), "Nigeria processor must be run to load labels"
        return geopandas.read_file(nigeria)[["lat", "lon"]]

    def labels_to_bounding_boxes(
        self, num_labelled_points: Optional[int], surrounding_metres: int
    ) -> List[EEBoundingBox]:

        output: List[EEBoundingBox] = []
        print(type(self.labels), self.labels)

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
        num_timesteps: int = 12,
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
    ) -> None:
        r"""
        Run the Nigeria exporter. For each label, the exporter will export

        int( (end_date - start_date).days / days_per_timestep) timesteps of data,

        where each timestep consists of a mosaic of all available images within the
        days_per_timestep of that timestep.

        :param days_per_timestep: The number of days of data to use for each mosaiced image.
        :param num_timesteps: The number of timesteps to export
        :param num_labelled_points: (Optional) The number of labelled points to export.
        :param surrounding_metres: The number of metres surrounding each labelled point to export
        :param checkpoint: Whether or not to check in self.data_folder to see if the file has
            already been exported. If it has, skip it
        :param monitor: Whether to monitor each task until it has been run
        """

        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            num_labelled_points=num_labelled_points,
            surrounding_metres=surrounding_metres,
        )

        start_date = self.data_date - num_timesteps * timedelta(days=days_per_timestep)

        for idx, bounding_box in enumerate(bounding_boxes_to_download):
            self._export_for_polygon(
                polygon=bounding_box.to_ee_polygon(),
                polygon_identifier=idx,
                start_date=start_date,
                end_date=self.data_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
            )
            