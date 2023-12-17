from pathlib import Path
import shutil
import pickle
from typing import List, Tuple, Dict, Union
from math import sin, cos, sqrt, atan2, radians

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#from src.engineer.nigeria import NigeriaDataInstance


class BoundedUniformSampler:
    # Adapted from https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/seco_downloader.py

    def __init__(self, country_shapefile_path):
        self.boundaries = gpd.read_file(country_shapefile_path).loc[0, 'geometry']

    def sample_point(self) -> List[List]:
        '''
        Returns lon, lan from within a country.
        '''
        minx, miny, maxx, maxy = self.boundaries.bounds
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        point = Point(lon, lat)
        if point.within(self.boundaries):
            return [point.x, point.y]
        else:
            return self.sample_point()
    
    def sample_n_points(self, n: int, buffer_km: float) -> gpd.GeoDataFrame:
        i = 0
        points = []
        while i < n:
            point = self.sample_point()
            too_close = self.check_distance_from_each_point(point[0], point[1], points, buffer_km=buffer_km)
            if not too_close:
                points.append(point)
                i += 1
        points = self.to_geodataframe(points)
        return points

    @staticmethod
    def to_geodataframe(points: List[List]) -> gpd.GeoDataFrame:
        '''Takes a list of list of lon/lat coordinates and converts it to a geopandas geodataframe.'''
        lon_lat = np.array(points)
        df = pd.DataFrame(lon_lat, columns=["lon", "lat"])
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
    
    @staticmethod
    def save_geodataframe(gdf: gpd.GeoDataFrame, path: Path, overwrite: bool = False) -> None:
        if not path.exists():
            gdf.to_file(path)
        else:
            print(f'File with path {path} already exists!')
            if overwrite:
                print('Overwriting file.')
                gdf.to_file(path)

    @staticmethod
    def distance_between_two_points_km(lon_lat1: Tuple, lon_lat2: Tuple) -> float:
        # https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
        R = 6373.0 # approximate radius of earth in km
        lon1 = radians(lon_lat1[0])
        lat1 = radians(lon_lat1[1])
        lon2 = radians(lon_lat2[0])
        lat2 = radians(lon_lat2[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    def check_distance_from_each_point(self, new_lon: float, new_lat: float, points: List[List[float]], buffer_km: float) -> bool:
        '''
        Returns True if new point is closer than the buffer distance (in km) from any point in points.
        ''' 
        for lon, lat in points:
            distance = self.distance_between_two_points_km(new_lon, new_lat, lon, lat)
            if distance < buffer_km:
                return True
        return False


distance_between_two_points_km = BoundedUniformSampler.distance_between_two_points_km


class BufferedResampler:

    subsets = ['training', 'validation', 'testing']

    def __init__(self, data_path: Path, val_set_size: float, test_set_size: float, buffer: float, engineer):
        self.data_path = data_path
        self.val_set_size = val_set_size
        self.test_set_size = test_set_size
        self.buffer = buffer
        self.engineer = engineer

    def read_labels(self) -> gpd.GeoDataFrame:
        rows = []
        for subset in self.subsets:
            pickle_files = [file for file in (self.data_path / subset).glob('*.pkl')]
            for file in pickle_files:
                identifier = file.name.split('_')[0]
                date = '_'.join(pickle_files[0].name.split('_')[1:]).split('.')[0]
                with file.open("rb") as f:
                    target_datainstance = pickle.load(f)
                #assert isinstance(target_datainstance, NigeriaDataInstance), 'Pickle file is not an instance of geowiki data'
                label = target_datainstance.is_crop

                rows.append((identifier,
                    date,
                    target_datainstance.instance_lat,
                    target_datainstance.instance_lon,
                    label,
                    file.name,
                    subset
                    ))

        df = pd.DataFrame(rows, columns=['identifier', 'date', 'lat', 'lon', 'label', 'filename', 'set'])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.lon, y=df.lat), crs='epsg:4326')
        return gdf

    @staticmethod
    def get_label_distribution(df: Union[pd.DataFrame, gpd.GeoDataFrame], column_to_groupby: str = 'set') -> pd.DataFrame:
        label_dist = df.groupby([column_to_groupby])['label'].agg(['count', 'sum'])
        label_dist['ratio'] = label_dist['sum'] / label_dist['count']
        label_dist.loc['total'] = [len(df), df['label'].sum(), df['label'].sum()/len(df)]
        label_dist.rename(columns={'sum': 'cropland_count'}, inplace=True)
        print('Label distribution:', label_dist)
        return label_dist
    
    @staticmethod
    def is_within_buffer_from_each_point(new_lon: float, new_lat: float, points_df: pd.DataFrame, buffer_km: float) -> bool:
        '''
        Returns True if new point is within the buffer distance (in km) from any point in points.
        ''' 
        for index, point in points_df.iterrows():
            lon, lat = point['lon'], point['lat']
            if new_lon == lon and new_lat == lat:
                return True
            distance = distance_between_two_points_km((new_lon, new_lat), (lon, lat))
            if distance < buffer_km:
                #print('too close!', distance)
                return True
        return False

    def sample_n_points_with_buffer(self, df: Union[pd.DataFrame, gpd.GeoDataFrame], n: int, buffer_km: float) -> pd.DataFrame:    
        i = 0
        points = pd.DataFrame()
        while i < n:
            point = df.sample(1)
            lon = float(point['lon'].values)
            lat = float(point['lat'].values)
            too_close = self.is_within_buffer_from_each_point(lon, lat, points, buffer_km)
            if not too_close:
                points = pd.concat([points, point])
                i += 1
        return points

    def resample_splits_with_buffer(self, gdf: gpd.GeoDataFrame):
        
        n_test = int(self.test_set_size * len(gdf))
        n_val = int(self.test_set_size * len(gdf))
        
        print("Resampling...")
        # Get test set
        test_set = self.sample_n_points_with_buffer(gdf, n_test, self.buffer)
        test_set['new_set'] = 'testing'
        
        # Get validation set
        train_val_set = gdf.iloc[gdf.index.difference(test_set.index)]
        val_set = self.sample_n_points_with_buffer(train_val_set, n_val, self.buffer)
        val_set['new_set'] = 'validation'

        # Get training set
        train_set = gdf.iloc[train_val_set.index.difference(val_set.index)]
        train_set['new_set'] = 'training'

        # Merge sets
        new_gdf = pd.concat([train_set, val_set, test_set])   
        assert len(gdf) == len(new_gdf), 'Number of samples is not preserved'

        return new_gdf

    def move_files(self, new_gdf: gpd.GeoDataFrame, backup_original: bool=True) -> None:
        backup_path = self.data_path / 'original_split'
        backup_original = False if backup_path.exists() else backup_original # only do it the first time
        if backup_original:
            # Copy original directory into a new subfolder 
            shutil.copytree(self.data_path, backup_path)

        for index, row in new_gdf.iterrows():
            source_path = self.data_path / row['set'] / row['filename']
            dest_path = self.data_path / row['new_set'] / row['filename']
            if source_path == dest_path:
                continue
            shutil.move(source_path, dest_path)

    def recalculate_normalizing_dict(self, dict_file_name: str = "normalizing_dict.pkl") -> Dict:
        dict_path = self.data_path / dict_file_name
        for subset in ['training', 'validation']:
            pickle_files = [file for file in (self.data_path / subset).glob('*.pkl')]
            for file_path in pickle_files:
                with file_path.open("rb") as f:   
                    target_datainstance = pickle.load(f)
                self.engineer.update_normalizing_values(target_datainstance.labelled_array)  

        normalizing_dict = self.engineer.calculate_normalizing_dict()
        with dict_path.open("wb") as f:
            print("Saved new normalizing dictionary:", dict_path)
            pickle.dump(normalizing_dict, f)

        return normalizing_dict

    def resample(self) -> None:
        gdf = self.read_labels() # new to read labels again because of new paths
        self.get_label_distribution(gdf)
        new_gdf = self.resample_splits_with_buffer(gdf)
        self.get_label_distribution(new_gdf, column_to_groupby='new_set')
        self.move_files(new_gdf)
        self.recalculate_normalizing_dict()
        new_gdf.to_file(self.data_path / 'nigeria_stratified_labelled_v1_splits_resampled.shp')


if __name__ == '__main__':

    here = Path(__file__).absolute().parent
    root = here.parent.parent
    path = root / 'assets' / 'nigeria_borders.shp'

    sampler = BoundedUniformSampler(path)

    N = 2000
    points = sampler.sample_n_points(N, buffer_km=15)
    sampler.save_geodataframe(points, root / 'data' / 'raw' / 'nigeria' / 'nigeria_stratified_v1.shp')