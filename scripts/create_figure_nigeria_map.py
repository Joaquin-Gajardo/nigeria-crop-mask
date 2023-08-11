import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
import rasterio
from rasterio.plot import show
import geopandas as gpd

sys.path.append('..')

from src.utils.misc import get_great_circle_distance


def main(version: int, map_type: str = 'binary') -> None:

    # Define tif path
    preds_dir = Path(f"../data/predictions/nigeria-cropharvest-full-country-2020/v{version}")
    tif_path = preds_dir / f'combined_{map_type}_uint8_lzw_clipped.tif'
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
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Nigeria borders
    borders = gpd.read_file(Path('../assets/nigeria_borders.shp'))
    borders.boundary.plot(ax=ax, color='grey', linewidth=0.5)
    
    # Plot map
    if map_type == 'binary':
        colors = ["wheat", "green"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([0, 1], 2)

        show(masked_arr, ax=ax, cmap=cmap, norm=norm, interpolation="nearest", transform=meta['transform'])

        # Legend
        legend_labels = {colors[1]: "cropland", colors[0]: "non-cropland"}
        patches = [Patch(facecolor=color, label=label, edgecolor="black") for color, label in legend_labels.items()]
        ax.legend(handles=patches, facecolor="white", fontsize=12, loc="lower right")

    elif map_type == 'probability':
        cmap = mpl.cm.get_cmap().copy()
        
        ax = show(masked_arr, ax=ax, cmap=cmap, interpolation="nearest", transform=meta['transform'])
        
        # Colorbar
        cbar = plt.colorbar(ax.get_images()[0], ax=ax, fraction=0.031) # need to get mappable from show() output, overloading name to ax probably not
        tick_labels = [f'{tick/100:.1f}' for tick in cbar.get_ticks()] # hack because converting data to float is too expensive
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('Cropland probability', fontsize=12)
    
    ## Common stuff ##

    # North arrow
    ax.text(x=3, y=13, s='N', fontsize=20, horizontalalignment='center')
    ax.arrow(3, 13 , 0, 0.01, width=0, length_includes_head=True, head_width=0.4, head_length=0.4, overhang=.3, facecolor='k')

    # Scale bar
    lat, lon = 9.042217, 7.288160 # random point in Nigeria
    dx = get_great_circle_distance(lat, lon)
    scalebar = ScaleBar(dx, "m", length_fraction=0.25, location="lower left", border_pad=1)
    ax.add_artist(scalebar)

    # Title and axes
    plt.title(f"Nigeria 2020 cropland {map_type} map", fontsize=16)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.minorticks_on()

    print(f'Saving figure to disk ...')
    plt.savefig(str(tif_path).replace('.tif', '.pdf'), dpi=300)  # TODO: save as pdf instead


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=2, help="Map version number, e.g. 0, 1 or 2")
    parser.add_argument("--map_type", type=str, default='binary', choices=['binary', 'probability'], help="Map type, either 'binary' or 'probability'")
    args = parser.parse_args()

    if args.map_type == 'both':
        for map_type in ['binary', 'probability']:
            main(args.version, map_type)
    else:
        main(args.version, args.map_type)