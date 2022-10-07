from pathlib import Path
import pickle

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from typing import Dict, List, Optional


class GDriveExporter:
    r"""
    An exporter to download data from Google Drive
    """

    scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]

    def __init__(self, data_folder: Path, dataset: str) -> None:
        self.data_folder = data_folder
        self.dataset = dataset
        self.raw_folder = self.data_folder / "raw"
        self.output_folder = self.raw_folder / self.dataset
        self.output_folder.mkdir(parents=True, exist_ok=True) # we will only save the token here

        assert (self.output_folder / "credentials.json").exists(), (
            f"Enable the google drive API at this link: "
            f"https://developers.google.com/drive/api/v3/quickstart/python "
            f"to use this class. Save the credentials.json at {self.output_folder}"
        )

        # https://developers.google.com/drive/api/v3/quickstart/python
        creds = None
        token_path = self.output_folder / "token.pickle"
        if token_path.exists():
            with token_path.open("rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.output_folder / "credentials.json", self.scopes
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with (self.output_folder / "token.pickle").open("wb") as token:
                pickle.dump(creds, token)

        self.service = build("drive", "v3", credentials=creds)

    def export(self, max_downloads: Optional[int] = None) -> None:
        r"""
        Download all tiffs from Google Drive folder with name specified in the self.dataset instance attribute.  

        :param max_downloads: The max number of downloads. If None, all tiff files containing
            region_name are downloaded
        """
        # Get parent folder ID
        folder_name = self.dataset
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

        print(f"Downloading {len(file_info)} tiff files from {folder_name} folder in Google Drive")

        for idx, individual_file in enumerate(file_info):
            if (max_downloads is not None) and (idx >= max_downloads):
                return None

            print(f"Downloading {individual_file['name']}")

            url = f"https://drive.google.com/uc?id={individual_file['id']}"

            download_path = (
                self.output_folder / individual_file["name"]
            )
            
            if download_path.exists():
                print(f"File already exists! Skipping")
                continue

            data = self.service.files().get_media(fileId=individual_file['id']).execute() # https://stackoverflow.com/questions/65053558/google-drive-api-v3-files-export-method-throws-a-403-error-export-only-support
            if data:
                with open(download_path, 'wb') as f:
                    f.write(data)

        print('Program finished')
