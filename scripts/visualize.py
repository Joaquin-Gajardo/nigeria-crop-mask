import sys
from pathlib import Path

import matplotlib.pyplot as plt
import geopandas as gpd

sys.path.append('..')

from src.engineer.nigeria import NigeriaEngineerNew
from src.utils.samplers import BufferedResampler


def plot_nigeria_dataset():

    data_path = Path('../data/features/nigeria/')
    engineer = NigeriaEngineerNew(Path('../data'))
    resampler = BufferedResampler(data_path, test_set_size=0.25, val_set_size=0.25, buffer=30, engineer=engineer)

    gdf = resampler.read_labels()
    nigeria = gpd.read_file(Path('../assets/nigeria_borders.shp'))

    gdf.replace('testing', 'test', inplace=True)
    gdf.replace('training', 'train', inplace=True)
    gdf.replace('validation', 'validation', inplace=True)

    fig, ax = plt.subplots(figsize=(20, 20))

    nigeria.boundary.plot(ax=ax, color='black')
    gdf.plot(ax=ax, column='set', markersize=30, cmap='viridis', legend=True, legend_kwds={'fontsize': 24, 'loc': 'lower right', 'markerscale': 2})

    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)

    ax.tick_params(labelsize=16)
    ax.minorticks_on()
    ax.tick_params(size=5, which='minor')

    fig.savefig('../figures/nigeria_dataset_splits.png', bbox_inches='tight', dpi=600)


plot_nigeria_dataset()