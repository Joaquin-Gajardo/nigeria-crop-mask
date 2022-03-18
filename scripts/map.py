import netCDF4 as nc
from scipy.fftpack import dst
import xarray
from rasterio.crs import CRS
from rasterio.merge import merge
import rioxarray
from rioxarray.merge import merge_datasets
from pathlib import Path
import sys

#cwd = Path.cwd()
file_path = '/home/gajo/code/togo-crop-mask/data/LandCoverMapper/preds_Togo_2019-04-26_2020-04-20-0000000000-000000000(1).tif.nc'
folder_path = Path('/home/gajo/code/togo-crop-mask/data/LandCoverMapper/')

#### 1. Convert .nc to .tif
## With netCDF4
# https://towardsdatascience.com/read-netcdf-data-with-python-901f7ff61648

# data = nc.Dataset(file_path)

# for dim in data.dimensions.values():
#      print(dim)
# for var in data.variables.values():
#      print(var)

# lat, lon = slice(0, 2), slice(0, 2)
# print(lat)

# print(data['lat'][lat])
# print(data['lon'][lon])
# print(data['prediction_0'][lat, lon])


## With xarray
# https://gis.stackexchange.com/questions/323317/converting-netcdf-dataset-array-to-geotiff-using-rasterio-python
file_paths = folder_path.glob('*.nc')

#file_paths_list = [path for path in file_paths]
xarrays_list = [(path, xarray.open_dataset(path)) for path in file_paths]
#xarrays_list = [(path, rioxarray.open_rasterio(path)) for path in file_paths] # alternatively rioxarray.open_rasterio could be used, but gives a xarray.DataArray as output and transform doesn't change

#print(sys.getsizeof(xarrays_list))
#print(len(xarrays_list))
#xarrays_list = xarrays_list[:2] # REMOVE. Just for testing purposes
#print(xarrays_list)
#xarrays_list = [xarr.rio.write_crs('epsg:4326') for xarr in xarrays_list]
#xarrays_list = [xarr.rename(lon='x', lat='y') for xarr in xarrays_list]

xarrays_list_new = []
tifs_list = []
tifs = False
dst_folder_path = Path('/home/gajo/DFS/Projects/Sims-gajo/projects/dataorg/togo_paper')
for path_in, xarr in xarrays_list:
    filename_out = '.'.join(path_in.name.split('.')[:-1])
    path_out = dst_folder_path / filename_out
    print(path_out)
    print(xarr)
    print(sys.getsizeof(xarr))
    print(xarr.rio.transform()) # No coordinate system
    new_xarr = xarr.rio.write_crs("epsg:4326")
    #new_xarr = new_xarr.rio.write_transform()
    print(new_xarr.rio.transform())
    new_xarr = new_xarr.rename(lon='x', lat='y')
    print(new_xarr.rio.transform())
    print(new_xarr.rio.crs)
    print(new_xarr.attrs) # --> shouldn't be that empty
    new_xarr = new_xarr.rio.reproject(new_xarr.rio.crs) # apply correct transformation
    print(new_xarr.rio.transform())
    print(new_xarr)
    #print(dir(new_xarr))
    xarrays_list_new.append(new_xarr)
    if tifs:
        new_xarr.rio.to_raster(path_out)
        tifs_list.append(path_out)
    #print(new_arr.spatial_ref)
#print(xarrays_list_new)
#print(tifs_list)

if tifs:
    #reimported_tif_test = rioxarray.open_rasterio(tifs_list[0])
    #print(reimported_tif_test)
    print(tifs_list)
    merged_tif = merge(tifs_list, dst_path=dst_folder_path/'merged_test_2.tif')
    ''' Error: --> with merge tifs, solved by hacking rasterio that is reading the wrong heights probably because of bounds problem
    File "/home/gajo/miniconda3/envs/togo-paper/lib/python3.6/site-packages/rasterio/merge.py", line 261, in merge
    dest = np.zeros((output_count, output_height, output_width), dtype=dt)
    ValueError: negative dimensions are not allowed
    '''
merged_xarr = merge_datasets(xarrays_list_new) # --> works but when writting tif sometimes it has 0 bytes. Could be because of incomplete bounds, because when doing all patches it works. Only problem is that resulting tiff file is 4.6 GB
# Error --> rasterio.errors.WindowError: Bounds and transform are inconsistent
print(type(merged_xarr), merged_xarr)
print(dir(merged_xarr))

folder_out = Path('/mnt/N/dataorg-datasets/MLsatellite/cropland-GEE-data/Togo/predictions/')
merged_xarr.rio.to_raster(folder_out / 'merged_xarr.tif')

sys.exit()

xds.rio.to_raster('./test.tif')


merged_xarr = merge_datasets(xarrays_list)
print(sys.getsizeof(merged_xarr), type(merged_xarr), merged_xarr)
# Error --> rasterio.errors.WindowError: Bounds and transform are inconsistent


xds = xarray.open_dataset(file_path)
print(sys.getsizeof(xds))
print(type(xds), xds)
xds.rio.write_crs("epsg:4326", inplace=True)
#xds.rio.set_spatial_dims("lon", "lat", inplace=True)
xds = xds.rename(lon='x', lat='y') #otherwise it complains that it doesn't find y
print(type(xds), xds)
xds.rio.to_raster('./test.tif')



#### 2. Merge all tiffs into one
# How to merge all xarray into one tiff? --> concat xarrays and then transform to tif

# Use rasterio.merge.merge like Danya: https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html
# Alternatively I could rioxarray.merge.merge_arrays, so I don't need to convert to tifs first. https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray-merge-module