import json
from pathlib import Path
import sys

import pandas as pd

here = Path(__file__).resolve().parent
ROOT = here.parent

def main(exp_name: str='final', model_name: str='lstm'):
    
    results_folder = 'results'
    path = ROOT / results_folder / exp_name / model_name
    if not path.exists():
        raise ValueError(f'Path {path} does not exist. Make sure you set up the correct exp_name (1st arg) and model_base (2nd arg). Args passed: {exp_name}, {model_name}')

    filepaths = list(path.glob('*.json'))

    results = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            result = json.load(f)
            result.update({'result_filepath': filepath})
            results.append(result)

    df = pd.DataFrame(results)

    # For results of multiheaded, rename column metric.
    # This is because test set is only on Nigeria so there won't be any global preds or labels,
    # only Nigeria_preds, and Nigeria_labels so output dict for pred and labels tensors will be empty

    columns_to_modify = [
    'test_roc_auc_score',
    'test_precision_score',
    'test_recall_score',
    'test_f1_score',
    'test_accuracy',
    'test_TN',
    'test_FP',
    'test_FN',
    'test_TP',
    ]

    columns_to_remove = [
    'test_Nigeria_roc_auc_score',
    'test_Nigeria_precision_score',
    'test_Nigeria_recall_score',
    'test_Nigeria_f1_score',
    'test_Nigeria_accuracy',
    'test_Nigeria_TN',
    'test_Nigeria_FP',
    'test_Nigeria_FN',
    'test_Nigeria_TP',
    ]

    for good_column, bad_column in zip(columns_to_modify, columns_to_remove):
        df[good_column].fillna(df[bad_column], inplace=True)
        del df[bad_column]

    # Reordering columns and rows
    col = df.pop('final_epoch')
    df.insert(6, col.name, col)

    df['result_timestamp'] = df['result_filepath'].apply(lambda p: p.stem)
    df = df.sort_values('result_timestamp').reset_index(drop=True)

    df.to_csv(path / f'results_{exp_name}_{model_name}.csv')

    return df


if __name__ == '__main__':
    
    assert len(sys.argv) == 3, 'This script takes 2 arguments: folder name with the experiment and the model name.'
    exp_name = sys.argv[1]
    model_name = sys.argv[2]

    df = main(exp_name, model_name)
    print(df)