import sys
from pathlib import Path
from datetime import date

import pandas as pd

sys.path.append("..") 

from src.exporters import (
    GeoWikiExporter,
    GeoWikiSentinelExporter,
    RegionalExporter,
    TogoSentinelExporter,
    NigeriaSentinelExporter,
    GDriveExporter,
    cancel_all_tasks,
)


def export_geowiki():

    exporter = GeoWikiExporter(Path("../data"))
    exporter.export()


def export_geowiki_sentinel_ee():
    exporter = GeoWikiSentinelExporter(Path("../data"))
    exporter.export_for_labels(num_labelled_points=None, monitor=False, checkpoint=True)


def export_togo():
    exporter = TogoSentinelExporter(Path("../data"))
    exporter.export_for_labels(
        num_labelled_points=None, monitor=False, checkpoint=True, evaluation_set=True
    )


def export_nigeria(): # TODO: fix GEE and gcloud API connection via CLI, currently only working to run this function via jupyter notebook
    exporter = NigeriaSentinelExporter(Path("../data"))
    exporter.export_for_labels(
        num_labelled_points=None, monitor=False, checkpoint=True
    )


def export_region():
    exporter = RegionalExporter(Path("../data"))
    exporter.export_for_region(
        region_name="Togo",
        end_date=date(2020, 4, 20),
        monitor=False,
        checkpoint=True,
        metres_per_polygon=None,
    )

def export_gdrive_nigeria():
    exporter = GDriveExporter(Path("../data"), dataset=NigeriaSentinelExporter.dataset)
    exporter.export_with_gdown()

def export_gdrive_nigeria_S1(api_key=None):
    # We need to put the credentials.json file in th output folder (get it from the google clound console). 
    # A token.pickle file will be generated on the same folder after authentification.
    exporter = GDriveExporter(Path('/media/Elements-12TB/satellite_images/nigeria'), dataset='nigeria-full-country-2020')
    path = Path('../gdrive_test_eo_data_files.csv')
    file_info = []
    if path.exists():
        print('Loading csv files with names')
        df = pd.read_csv(path)
        file_info = df.to_dict('records')
    else:
        print('Listing files on gdrive...')
        file_info = exporter.list_files_in_folder(folder_name='test_eo_data')
        df = pd.DataFrame(file_info)
        df.to_csv(path, index=False)
    
    # You can generate an API key on the google cloud console
    exporter.export_with_api_key(file_info, api_key)

def cancel_tasks():
    cancel_all_tasks()


if __name__ == "__main__":
    #export_geowiki()
    #export_geowiki_sentinel_ee()
    #export_togo() # --> why is the default only for evaluation set?
    #export_region()

    #export_nigeria()
    #export_gdrive_nigeria()
    export_gdrive_nigeria_S1()
        
    ## Original ##
    #export_geowiki_sentinel_ee()
    #export_togo()
    #export_region()
    ##cancel_all_tasks()
