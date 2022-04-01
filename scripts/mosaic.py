import xarray
import rioxarray
from rioxarray.merge import merge_datasets

import argparse
from functools import cache
from pathlib import Path

@cache
def merge_netcdf4(input_folder: str, output_path: str, save_as_tif: bool=True) -> xarray.Dataset: 
    file_paths = Path(input_folder).glob('*.nc')
    xarrays_list = [xarray.open_dataset(path) for path in file_paths] # alternatively rioxarray.open_rasterio

    xarrays_list_new = []
    for xarr in xarrays_list:
        new_xarr = xarr.rio.write_crs("epsg:4326")
        new_xarr = new_xarr.rename(lon='x', lat='y')
        new_xarr = new_xarr.rio.reproject(new_xarr.rio.crs) # apply correct transformation
        xarrays_list_new.append(new_xarr)

    merged_xarr = merge_datasets(xarrays_list_new)
    print(merged_xarr)
    merged_xarr.to_netcdf(output_path.split('.')[0] + '.nc')
    if save_as_tif:
        print('Saving.')
        merged_xarr.rio.to_raster(Path(output_path))
        print(f'Tif saved to: {output_path}.')

    # TODO: add compressing within script
    # gdal_translate /mnt/N/dataorg-datasets/MLsatellite/cropland-GEE-data/Togo/predictions/merged_xarr.tif /mnt/N/dataorg-datasets/MLsatellite/cropland-GEE-data/Togo/predictions/merged_xarr_deflate.tif -co COMPRESS=DEFLATE -co PREDICTOR=3

    return merged_xarr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str, default='/home/gajo/code/togo-crop-mask/data/predictions/Nigeria/LandCoverMapper/')
    parser.add_argument('--output-path', type=str, default='/mnt/N/dataorg-datasets/MLsatellite/cropland-GEE-data/Nigeria/predictions/merged_xarr_trainedTogo_18032022.tif')
    args = parser.parse_args()

    _ = merge_netcdf4(args.input_folder, args.output_path)