"""
This script will create a vrt file for each nc file in the predictions directory
and then merge them into a single .tif file.

Code largely taken from https://github.com/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/create_map.ipynb
"""

import os
import sys
from pathlib import Path
import warnings

from tqdm import tqdm

sys.path.append('..')
warnings.filterwarnings("ignore", message="Warning 1: No UNIDATA NC_GLOBAL:Conventions attribute")


def build_vrt(preds_dir, filename = 'combined'):
    ncs_dir = preds_dir / "nc_files"
    vrts_dir = preds_dir / "vrt_files"
    vrts_dir.mkdir(exist_ok=True, parents=True)    

    # Create vrt files for each nc file
    nc_files = list(ncs_dir.glob('*.nc'))
    for nc_file_path in tqdm(nc_files):
        identifier = nc_file_path.stem.split('_')[1]
        vrt_file_path = vrts_dir / f'{identifier}.vrt'
        if not vrt_file_path.exists():
            print(f'Creating vrt {vrt_file_path.name}')
            os.system(f'gdalbuildvrt {str(vrt_file_path)} {str(nc_file_path)}')

    # Combine all vrt files into one
    combined_vrt_path = preds_dir / filename / '.vrt' 
    if not combined_vrt_path.exists():
        print(f'Merging vrt files into {combined_vrt_path} ...')
        os.system(f'gdalbuildvrt {str(combined_vrt_path)} {str(vrts_dir / "*.vrt")} ')

def build_tif(preds_dir, vrt_filename, tif_filename):
    vrt_path = preds_dir / f'{vrt_filename}.vrt'
    tif_path = preds_dir / f'{tif_filename}.tif'

    if tif_path.exists():
        print(f'{tif_path} already exists, exiting!')
        return
    if not vrt_path.exists():
        build_vrt(preds_dir, vrt_filename)

    print(f'Converting {vrt_filename}.vrt to {tif_filename}.tif ...')
    os.system(f'gdal_translate -a_srs EPSG:4326 -of GTiff {str(vrt_path)} {str(tif_path)}') # -co "COMPRESS=LZW" -co "TILED=YES"')

def main():
    preds_dir = Path("../data/predictions/nigeria-cropharvest-full-country-2020")
    vrt_filename = 'combined'
    tif_filename = f'{vrt_filename}'
    build_tif(preds_dir, vrt_filename, tif_filename)


if __name__ == '__main__':
    main()

