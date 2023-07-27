import os
from pathlib import Path
import sys
from datetime import date

from cropharvest.inference import Inference

sys.path.append('..')

from src.models import LandCoverMapper


def main(start_stop):
    start, stop = start_stop

    test_folder = Path("/media/Elements-12TB/satellite_images/nigeria/raw/nigeria-full-country-2020")
    preds_dir = Path("/media/Elements-12TB/satellite_images/nigeria/predictions/nigeria-full-country-2020")
    model_path = "../data/lightning_logs/version_893/checkpoints/epoch=25.ckpt" # obtained with python models.py --max_epochs 35 --train_with_val True --inference True --geowiki_subset neighbours1

    preds_dir.mkdir(exist_ok=True, parents=True)

    test_files = sorted(test_folder.glob("*.tif"), key=lambda x:int(x.stem.split('-')[0]))
    
    ## Need to include start_date on the tif filenames otherwise Inference.run complains. Do it only once.
    # start_date = date(2019, 4, 3).strftime('%Y-%m-%d')
    # for path in test_files:
    #     new_filename = f'{path.stem}_{start_date}{path.suffix}'
    #     new_path = path.parent / new_filename
    #     path.replace(new_path)
    # test_files = sorted(test_folder.glob("*.tif"), key=lambda x:int(x.stem.split('-')[0]))

    assert all([file.exists() for file in test_files])

    model = LandCoverMapper.load_from_checkpoint(model_path)

    inferer = Inference(model=model, normalizing_dict=None, batch_size=8192)

    skips_filename = 'skipped_files.txt'
    warnings_filename = 'warning_files.txt'

    for i, path in enumerate(test_files):
        
        if (start <= i < stop):
            print(f'Predicting on file {path}')
            try:
                preds = inferer.run(path, dest_path=preds_dir / f"preds_{path.name}.nc")

                if preds.min() <= 0 or preds.max() >= 1:
                    Warning(f'preds for file {path.name} are not between 0 and 1!')
                    os.system(f'echo {str(path)} >> {str(preds_dir / warnings_filename)}')
            
            except RuntimeError as e:
                print('Encounter the following error, skipping!: ', e)
                os.system(f'echo {str(path)} >> {str(preds_dir / skips_filename)}')
                continue
        else:
            continue


if __name__ == '__main__':
    
    #main()

    ### Multiprocessing for predictions ###

    from multiprocessing import Pool

    workers = 20
    starts = list(range(0, 14000, 710))
    stops = list(range(710, 14201, 710))
    start_stop_indices = list(zip(starts, stops))
    assert workers == len(start_stop_indices)

    with Pool(workers) as p:
        print(p.map(main, start_stop_indices))

