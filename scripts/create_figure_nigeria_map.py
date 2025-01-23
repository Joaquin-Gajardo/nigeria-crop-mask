import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.plot import show

sys.path.append('..')

from create_map_nigeria import COMPRESSION, DTYPE, MAP_TYPES, MAP_VERSION

from src.utils.misc import get_great_circle_distance


def main(version: int, map_type: str = 'binary') -> None:

    # Define tif path
    preds_dir = Path(f"../data/predictions/nigeria-cropharvest-full-country-2020/v{version}")
    tif_path = preds_dir / f'combined_{map_type}_{DTYPE}_clipped_{COMPRESSION}.tif'
    assert tif_path.exists(), f'{tif_path} does not exist, make sure the arguments are correct!'

    print(f'Generating {map_type} map figure from {tif_path} ...')

    # Read tif #TODO: read with rioxarray instead
    with rasterio.open(tif_path) as src:
        meta = src.meta
        data = src.read(1)
    nodata_value = int(meta['nodata'])

    # Mask array
    masked_arr = np.ma.array(data, mask=(data == int(nodata_value)))
    del data

    ### Plot ###
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Plot Nigeria country and state borders
    borders = gpd.read_file(Path('../assets/nigeria_borders.shp'))
    borders.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    # nigeria_states = gpd.read_file('../assets/ngaadmbndaadm1osgof20161215.geojson')
    # nigeria_states.to_crs(borders.crs, inplace=True)
    # nigeria_states.boundary.plot(ax=ax, color='grey', linewidth=0.5)
    
    # Plot map
    if map_type == 'binary':
        colors = ["wheat", "green"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([0, 1], 2)

        show(masked_arr, ax=ax, cmap=cmap, norm=norm, interpolation="nearest", transform=meta['transform'])
        ax.annotate("A", xy=(0.02, 0.95), xycoords='axes fraction', fontsize=34, fontweight='bold')

        # Legend
        legend_labels = {colors[1]: "cropland", colors[0]: "non-cropland"}
        patches = [Patch(facecolor=color, label=label, edgecolor="black") for color, label in legend_labels.items()]
        ax.legend(handles=patches, facecolor="white", fontsize=20, loc="lower right")

    elif map_type == 'probability':
        cmap = mpl.cm.get_cmap().copy()
        
        ax = show(masked_arr, ax=ax, cmap=cmap, interpolation="nearest", transform=meta['transform'])
        ax.annotate("B", xy=(0.02, 0.95), xycoords='axes fraction', fontsize=34, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(ax.get_images()[0], ax=ax, fraction=0.031) # need to get mappable from show() output, overloading name to ax probably not
        tick_labels = [f'{tick/100:.1f}' for tick in cbar.get_ticks()] # hack because converting data to float is too expensive
        cbar.set_ticklabels(tick_labels, fontsize=16)
        cbar.set_label('Cropland probability', fontsize=20)
    
    ## Common stuff ##

    # North arrow
    ax.annotate('N', xy=(0.95, 0.92), fontsize=34, xycoords='axes fraction', horizontalalignment='center', verticalalignment='bottom')
    ax.arrow(0.95, 0.92, 0, 0.01, length_includes_head=True, head_width=0.03, head_length=0.03, overhang=.3, facecolor='k', transform=ax.transAxes)

    # Scale bar
    lat, lon = 9.042217, 7.288160 # random point in Nigeria
    dx = get_great_circle_distance(lat, lon)
    scalebar = ScaleBar(dx, "m", length_fraction=0.25, location="lower left", border_pad=1, font_properties={'size': 16})
    ax.add_artist(scalebar)

    # Title and axes
    plt.title(f"Nigeria 2020 cropland {map_type} map", fontsize=24)
    ax.tick_params(labelsize=16)
    plt.xlabel("Longitude (°)", fontsize=20)
    plt.ylabel("Latitude (°)", fontsize=20)
    plt.minorticks_on()
    ax.set_xlim(2, 15.2)
    ax.set_ylim(3.7, 14.3)

    print(f'Saving figure to disk ...')
    plt.savefig(str(tif_path).replace('.tif', '_latest.pdf'), dpi=300)


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("--version", type=int, default=4, help="Map version number, e.g. 0, 1, 2, etc")
    parser.add_argument("--map_type", type=str, default='binary', choices=MAP_TYPES + ['both'], help=f"Map type, either {MAP_TYPES}, or 'both'")
    args = parser.parse_args()

    if args.map_type == 'both':
        for map_type in MAP_TYPES:
            main(MAP_VERSION, map_type)
    else:
        main(MAP_VERSION, args.map_type)