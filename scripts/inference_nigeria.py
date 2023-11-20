import multiprocessing
import os
import sys
from datetime import date
from pathlib import Path

from cropharvest.inference import Inference

sys.path.append('..')

from src.models import LandCoverMapper


def main(start_stop=(0, None)):
    start, stop = start_stop

    dataset_name = 'nigeria-cropharvest-full-country-2020'
    map_version = 4
    raw_folder = Path(f"/media/Elements/satellite_images/nigeria/raw/{dataset_name}")
    preds_dir = Path(f"../data/predictions/{dataset_name}/v{map_version}/nc_files")
    preds_dir.mkdir(exist_ok=True, parents=True)

    #model_path = "../data/lightning_logs/version_893/checkpoints/epoch=25.ckpt" # Map version 0. Model obtained with python models.py --max_epochs 35 --train_with_val True --inference True --geowiki_subset neighbours1
    #model_path = '../data/lightning_logs/version_896/checkpoints/epoch=21.ckpt' # Map version 1. Model obtained with python models.py --geowiki_subset neighbours1 --weighted_loss_fn --inference True
    #model_path = '../data/lightning_logs/version_899/checkpoints/epoch=21.ckpt' # Map version 2. Model obtained from the best results of bash run_experiments.sh final lstm 64 1 0.2 2 100 False True, which was --geowiki_subset neighbours1 --weighted_loss_fn --inference True
    #model_path = '../data/lightning_logs/version_949/checkpoints/epoch=22.ckpt' # Map version 3. Model obtained from the best results of bash run_experiments.sh final lstm 64 1 0.2 2 100 False True, which was geowiki_subset nigeria --weighted_loss_fn --inference True
    model_path = '../data/lightning_logs/version_949/checkpoints/epoch=22.ckpt' # Map version 4 (normalization as with test set). Model obtained from the best results of bash run_experiments.sh final lstm 64 1 0.2 2 100 False True, which was geowiki_subset nigeria --weighted_loss_fn --inference True

    
    raw_files = sorted(raw_folder.glob("*.tif"), key=lambda x:int(x.stem.split('-')[0]))
    pred_files = list(preds_dir.glob('*.nc'))

    ## Need to include start_date on the tif filenames otherwise Inference.run complains. Do it only once.
    # start_date = date(2019, 4, 3).strftime('%Y-%m-%d')
    # for path in raw_files:
    #     new_filename = f'{path.stem}_{start_date}{path.suffix}'
    #     new_path = path.parent / new_filename
    #     path.replace(new_path)
    # raw_files = sorted(raw_folder.glob("*.tif"), key=lambda x:int(x.stem.split('-')[0]))

    #assert all([file.exists() for file in raw_files])
    
    raw_files_indices = [path.stem.split('_')[0] for path in raw_files]
    pred_files_indices = [path.stem.split('_')[1] for path in pred_files]

    missing_files = list(set(raw_files_indices) - set(pred_files_indices))

    file_ending = raw_files[0].suffix
    date = raw_files[0].stem.split('_')[-1]

    missing_preds_paths = [raw_folder / f'{identifier}_{date}{file_ending}' for identifier in missing_files]
    missing_preds_paths = sorted(missing_preds_paths, key=lambda x:int(x.stem.split('-')[0]))
    
    model = LandCoverMapper.load_from_checkpoint(model_path, data_folder='../data')
    if model.training:
        model.eval()
    assert not model.training, 'Need to put model in eval mode, else will have problems with dropout'
    inferer = Inference(model=model, normalizing_dict=model.normalizing_dict, batch_size=8192)

    skips_filename = 'skipped_files.txt'
    warnings_filename = 'warning_files.txt'

    stop = len(missing_preds_paths) if stop is None else stop
    for i, path in enumerate(missing_preds_paths):
        
        if (start <= i < stop):

            dest_path = preds_dir / f"preds_{path.name}.nc"

            if dest_path.exists():
                print(f'{multiprocessing.current_process()}: file {dest_path} exists, skipping!')
            else:
                print(f'{multiprocessing.current_process()}: predicting on file {path}')
                try:
                    preds = inferer.run(path, dest_path=dest_path)
                    if preds.prediction_0.to_numpy().min() < 0 or preds.prediction_0.to_numpy().max() > 1:
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

    workers = 10
    # starts = list(range(0, 700, 100))
    # stops = list(range(100, 800, 100))
    starts = list(range(0, 13501, 1500))
    stops = list(range(1500, 15001, 1500))
    start_stop_indices = list(zip(starts, stops))
    print(start_stop_indices)
    assert workers == len(start_stop_indices)

    with Pool(workers) as p:
        print(p.map(main, start_stop_indices))

