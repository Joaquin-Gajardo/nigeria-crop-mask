"""
This script will generate an create a vrt file for each nc file in the predictions directory
and then merge them into a single .tif file. Then it will create both a binary map
and a cropland probability. Both output maps will be compressed, clipped to Nigeria,
stored as uint8 data type.

Hardware requirements: about 150GB of RAM and 64GB of free disk space (for intermediate files). Final output is about 5GB.

Code largely taken from https://github.com/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/create_map.ipynb
"""

import os
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm


DTYPE = rasterio.uint8
NODATA_VALUE = 255 
COMPRESSION = 'lzw'
MAP_TYPES = ['binary', 'probability']


def build_vrt(preds_dir: Path, filename: str = 'combined') -> None:
    ncs_dir = preds_dir / "nc_files"
    vrts_dir = preds_dir / "vrt_files"
    vrts_dir.mkdir(exist_ok=True, parents=True)    

    # Create invdividual vrt files for each nc file
    nc_files = list(ncs_dir.glob('*.nc'))
    vrt_files = list(vrts_dir.glob('*.vrt'))
    if len(nc_files) != len(vrt_files):
        for nc_file_path in tqdm(nc_files):
            identifier = nc_file_path.stem.split('_')[1]
            vrt_file_path = vrts_dir / f'{identifier}.vrt'
            if not vrt_file_path.exists():
                print(f'Creating vrt {vrt_file_path.name}')
                os.system(f'gdalbuildvrt {str(vrt_file_path)} {str(nc_file_path)}')

    # Combine all vrt files into one
    combined_vrt_path = preds_dir / f'{filename}.vrt' 
    if not combined_vrt_path.exists():
        print(f'Merging vrt files into {combined_vrt_path} ...')
        os.system(f'gdalbuildvrt {str(combined_vrt_path)} {str(vrts_dir / "*.vrt")} ')


def build_raw_tif(preds_dir: Path, filename: str = 'combined') -> Path:
    vrt_path = preds_dir / f'{filename}.vrt'
    raw_tif_path = preds_dir / f'{filename}.tif'

    if raw_tif_path.exists():
        print(f'{raw_tif_path} already exists!')
    else:
        if not vrt_path.exists():
            build_vrt(preds_dir, filename)

        print(f'Converting {vrt_path.name} to {raw_tif_path.name} ...')
        os.system(f'gdal_translate -a_srs EPSG:4326 -of GTiff {str(vrt_path)} {str(raw_tif_path)}')

    return raw_tif_path


def create_individual_map(input_tif_path: Path, output_tif_path: Path, borders_geometry, map_type: str) -> None:
    assert input_tif_path.exists(), f'{input_tif_path} does not exist!'
    assert map_type in ['binary', 'probability'], f'Invalid map type {map_type}. Must be one of ["binary", "probability"]'
    
    print(f'Creating {map_type} map ...')
    tmp_tif_path = Path(str(output_tif_path).replace('_clipped.tif', '.tif'))
    
    # Create intermediate file (change dtype and compress)
    if not tmp_tif_path.exists(): 
        # Read raw tif into memory
        print('Reading raw data into memory ...')
        with rasterio.open(input_tif_path) as src:
            meta = src.meta
            data = src.read(1)
        
        # Convert to binary data retaining nan values
        print('Processing ...')
        nan_mask = np.isnan(data)
        if map_type == 'binary':
            data = (data > 0.5).astype(DTYPE)
        elif map_type == 'probability':
            print(f'{map_type.capitalize()} map is converted to 0-100 scale')
            data = (data * 100).astype(DTYPE)
        else:
            # This should never be reached given the assertions above
            raise ValueError(f'Invalid map type {map_type}. Must be one of ["binary", "probability"]')

        data[nan_mask] = NODATA_VALUE

        # Save intermediate file to disk
        print(f'Saving intermediate file {tmp_tif_path.name} to disk ...')
        meta.update({'compress': COMPRESSION, 'dtype': DTYPE, 'nodata': NODATA_VALUE})
        with rasterio.open(tmp_tif_path, 'w', **meta) as dst:
            dst.write(data, 1)

    # Read intermediate file and clip to Nigeria
    print('Reading intermediate data into memory ...')
    with rasterio.open(tmp_tif_path) as src:
        meta = src.meta
        data, transform = mask(src, borders_geometry, crop=True, all_touched=True, nodata=NODATA_VALUE)

    # Save final file to disk
    print(f'Saving final file {output_tif_path.name} to disk ...')
    data = np.squeeze(data) # seems to save space in the final tif
    meta.update({"height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "compress": COMPRESSION})
    
    with rasterio.open(output_tif_path, "w", **meta) as dst:
        dst.write(data, 1)

    # Remove intermediate tif file from disk
    #os.remove(str(tmp_tif_path))


def create_maps(preds_dir: Path, base_filename: str = 'combined') -> None:

    # Read Nigeria borders
    borders_file = gpd.read_file(Path('../assets/nigeria_borders.shp'))
    borders_geometry = borders_file.geometry[0]

    # Target paths
    target_map_paths = {
        map_type: preds_dir / f'{base_filename}_{map_type}_{DTYPE}_{COMPRESSION}_clipped.tif' for map_type in MAP_TYPES
        }
    if all([target_map_path.exists() for target_map_path in target_map_paths.values()]):
        print('Both target maps already exist!')
        return
    
    # Build raw tif
    raw_tif_path = build_raw_tif(preds_dir, base_filename)

    # Create binary and probability maps
    for map_type in MAP_TYPES:
        target_tif_path = target_map_paths[map_type]
        if target_tif_path.exists():
            print(f'{map_type.capitalize()} map already exists as {target_tif_path.name} ! ')
        else:    
            create_individual_map(raw_tif_path, target_tif_path, borders_geometry, map_type)
    
    # Remove raw tif from disk and vrt files
    os.remove(str(raw_tif_path))
    os.remove(str(raw_tif_path).replace('.tif', '.tif.aux.xml'))
    #os.remove(str(raw_tif_path).replace('.tif', '.vrt'))
    #os.system(f'rm -r {str(preds_dir / "vrt_files")}')


def main(version: str) -> None:

    preds_dir = Path(f"../data/predictions/nigeria-cropharvest-full-country-2020/v{version}")
    if not preds_dir.exists():
        print(f'{preds_dir} does not exist, exiting!')
        return
    
    # Create maps from raw tif
    base_filename = 'combined'
    create_maps(preds_dir, base_filename)


if __name__ == '__main__':

    assert len(sys.argv) == 2, "Provide the map version number as an argument, e.g. 0, 1 or 2"
    version = sys.argv[1]
    assert version.isdigit(), "Version must be an integer."
    
    main(version)

