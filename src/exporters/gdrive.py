from pathlib import Path
import pickle
import os
import io

import pandas as pd
import gdown
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from typing import Dict, List, Optional


class GDriveExporter:
    r"""
    An exporter to download data from Google Drive
    """

    #scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    scopes = ["https://www.googleapis.com/auth/drive"]

    def __init__(self, data_folder: Path, dataset: str) -> None:
        self.data_folder = data_folder
        self.dataset = dataset
        self.raw_folder = self.data_folder / "raw"
        self.output_folder = self.raw_folder / self.dataset
        self.output_folder.mkdir(parents=True, exist_ok=True) # we will only save the token here

        credentials_path = self.output_folder / "credentials.json"
        assert credentials_path.exists(), (
            f"Enable the google drive API at this link: "
            f"https://developers.google.com/drive/api/v3/quickstart/python "
            f"to use this class. Save the credentials.json at {self.output_folder}"
        )

        # https://developers.google.com/drive/api/v3/quickstart/python
        creds = None
        token_path = self.output_folder / "token.json"
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(token_path, self.scopes)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.scopes)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        self.service = build("drive", "v3", credentials=creds)

    def list_files_in_folder(self, folder_name: str, file_path: Path=None, max_downloads: Optional[int] = None) -> None:
        r"""
        List all tiffs from a Google Drive folder or load file list from csv file..

        :param folder_name: list all tiffs in this google drive folder. Write list to csv file.
        :param file_path: Optional path where to write csv file or to read from, if not provided it will write it to 
        :param max_downloads: The max number of downloads. If None, all tiff files containing
            region_name are downloaded
        """
        file_info = []

        if file_path is None:
            file_path = self.output_folder / f"gdrive_{folder_name}_files.csv" 

        if file_path.exists():
            print(f'Loading of files names and ids from csv at {file_path}')
            df = pd.read_csv(file_path)
            file_info = df.to_dict('records')
            return file_info

        ## If file doesn't exist list all files in gdrive target folder and save to disk

        # Get parent folder ID
        print('Listing files on gdrive...')
        query = f'(fullText contains "{folder_name}") and (mimeType = "application/vnd.google-apps.folder")'

        folder_info: List[Dict] = [] 
        results = (
            self.service.files()
            .list(pageSize=10, q=query, fields="nextPageToken, files(id, name)",)
            .execute()
        )
        items = results.get("files", [])

        folder_info.extend(items) 
        assert len(folder_info) == 1, 'Should be a unique folder'

        folder_id = folder_info[0]['id']

        # Download all tiffs in the folder (by iterating over pages)
        query = f'("{folder_id}" in parents) and (mimeType = "image/tiff")'

        file_info: List[Dict] = []

        results = (
        self.service.files()
        .list(pageSize=10, q=query, fields="nextPageToken, files(id, name)",)
        .execute()
        )
        items = results.get("files", [])

        file_info.extend(items)

        next_page = results.get("nextPageToken", None)

        while next_page is not None:
            results = (
                self.service.files()
                .list(
                    pageSize=10,
                    pageToken=next_page,
                    # https://stackoverflow.com/questions/47402545/
                    # google-drive-js-api-nextpagetoken-invalid
                    q=query,
                    fields="nextPageToken, files(id, name)",
                )
                .execute()
            )

            items = results.get("files", [])
            file_info.extend(items)

            next_page = results.get("nextPageToken", None)

        print(f"There are {len(file_info)} tiff files from {folder_name} folder in Google Drive")

        # Write files to disk
        print(f'Writing file names and ids to {file_path}')
        df = pd.DataFrame(file_info)
        df.to_csv(file_path, index=False)

        return file_info

    def export(self, folder_name: str, reversed_order: bool=False) -> None:
        
        file_info = self.list_files_in_folder(folder_name)
        if reversed_order:
            file_info = reversed(file_info)

        for i, file_metadata in enumerate(file_info):
            
            file_id = file_metadata['id']
            file_name = file_metadata['name'].split('/')[-1]
            output_path = self.output_folder / file_name

            if output_path.exists():
                print(f"File {file_name} already exists! Skipping")
                continue
            else:
                print(f"Downloading file {i}/{len(file_info)} {file_name} with id {file_id} from drive into {output_path}")
                try:
                    # Read file by chunks
                    request = self.service.files().get_media(fileId=file_id)
                    data = request.execute()
                    if data:
                        with open(output_path, 'wb') as f:
                            print(f'Writing file to {output_path}')
                            f.write(data)    

                    # file = io.BytesIO()
                    # downloader = MediaIoBaseDownload(file, request)
                    # done = False
                    # while done is False:
                    #     status, done = downloader.next_chunk()
                    #     print(F'Download {int(status.progress() * 100)}.')
                    
                    # # Write to file
                    # with io.open(output_path, 'wb') as f:
                    #     print(f'Writing file to {output_path}')
                    #     file.seek(0)
                    #     f.write(file.read())            
        
                except HttpError as error:
                    print(F'An error occurred: {error}')
                    break

    def export_with_gdown(self, file_info: List[Dict]) -> None:

        if file_info is not None:
            for i, file in enumerate(file_info):
                file_id = file['id']
                file_name = file['name']
                output_path = self.output_folder / file_name.split('/')[-1]

                if output_path.exists():
                    print("File already exists! Skipping")
                    continue

                print(f"Downloading file {i}/{len(file_info)} {file_name} with id {file_id} from drive into {output_path}")
                gdown.download(id=file_id, output=str(output_path), quiet=False)

    def export_with_api_key(self, file_info: List[Dict], api_key: str) -> None:
        """Hits solving captcha issue after a few requests."""
        for i, file in enumerate(file_info):
            file_id = file['id']
            url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}'
            
            file_name = file['name'].split('/')[-1]
            output_path = self.output_folder / file_name

            if output_path.exists():
                print(f"File {file_name} already exists! Skipping")
                continue
            else:
                print(f"Downloading file {i}/{len(file_info)} {file_name} with id {file_id} from drive into {output_path}")
                os.system(f"curl -o {output_path} '{url}'")
            


