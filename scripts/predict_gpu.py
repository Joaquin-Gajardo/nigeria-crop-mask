# conda activate nigeria-crop-masl-gpu

r"""
A script to run and save predictions
"""
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from tqdm import tqdm

import sys

sys.path.append("..")

from src.models import LandCoverMapper
from src.engineer.base import BaseEngineer
from src.utils import sentinel_as_tci


def landcover_mapper():

    #data_dir = "../data"
    data_dir = "../data/predictions/Nigeria2"

    #test_folder = Path("/mnt/N/dataorg-datasets/MLsatellite/cropland-GEE-data/Togo/tifs/earth_engine_region/Takeout/Drive/earth_engine_region")
    test_folder = Path("/mnt/N/dataorg-datasets/MLsatellite/cropland-GEE-data/Nigeria/full_country/tifs/raw/gdrive")
    test_files = sorted(test_folder.glob("*.tif"), key=lambda x:int(x.stem.split('-')[0]))

    #model_path = "lightning_logs/version_1/checkpoints/epoch=2.ckpt" # cpu example: nigeria-crop-mask env
    model_path = "../data/lightning_logs/version_587/checkpoints/epoch=3.ckpt" # gpu example: nigeria-crop-mask-gpu env. Make sure model was run in this environment!

    print(f"Using model {model_path}")
    model = LandCoverMapper.load_from_checkpoint(model_path) # BUG: problem with loading state (hparams) in GPU environment, probably due to difference in lightning (0.7.1 vs 0.8.5) --> Solved by wrapping loaded hparams dict into Namespace

    for test_path in tqdm(test_files):

        print(f"Running for {test_path}")
        out = model.predict(test_path, use_gpu=True, batch_size=8192) # bigger batch size means we can paralleize more, should only affect speed as there are no gradients being calculated

        # the date passed is not too important here
        tci = sentinel_as_tci(
            BaseEngineer.load_tif(
                test_path, start_date=datetime(2020, 1, 1), days_per_timestep=30
            ),
            scale=False,
        ).isel(time=-1)

        tci = tci.sortby("x").sortby("y")
        out = out.sortby("lat").sortby("lon")

        plt.clf()
        fig, ax = plt.subplots(
            1, 3, figsize=(20, 7.5), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        fig.suptitle(
            f"Model results for tile with bottom left corner:"
            f"\nat latitude {float(out.lat.min())}"
            f"\n and longitude {float(out.lon.min())}",
            fontsize=15,
        )
        # ax 1 - original
        img_extent_1 = (tci.x.min(), tci.x.max(), tci.y.min(), tci.y.max())
        img = np.clip(np.moveaxis(tci.values, 0, -1), 0, 1)

        ax[0].set_title("True colour image")
        ax[0].imshow(
            img, origin="upper", extent=img_extent_1, transform=ccrs.PlateCarree()
        )

        # ax 2 - mask
        ax[1].set_title("Mask")
        im = ax[1].imshow(
            out.prediction_0,
            origin="upper",
            extent=img_extent_1,
            transform=ccrs.PlateCarree(),
            vmin=0,
            vmax=1,
        )

        # finally, all together
        ax[2].set_title("Mask on top of the true colour image")
        ax[2].imshow(
            img, origin="upper", extent=img_extent_1, transform=ccrs.PlateCarree()
        )
        ax[2].imshow(
            out.prediction_0 > 0.5,
            origin="upper",
            extent=img_extent_1,
            transform=ccrs.PlateCarree(),
            alpha=0.3,
            vmin=0,
            vmax=1,
        )

        fig.colorbar(
            im, ax=ax.ravel().tolist(),
        )

        save_dir = Path(data_dir) / model.__class__.__name__
        save_dir.mkdir(exist_ok=True, parents=True)

        plt.savefig(
            save_dir / f"results_{test_path.name}.png", bbox_inches="tight", dpi=300
        )
        plt.close()
        # plt.show()
        out.to_netcdf(save_dir / f"preds_{test_path.name}.nc")


if __name__ == "__main__":
    landcover_mapper()
