import torch
import sys
from argparse import ArgumentParser, Namespace

sys.path.append("..")

from src.models import STR2MODEL, train_model


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)

    model_args = STR2MODEL["land_cover"].add_model_specific_args(parser).parse_args()
    print('Default model arguments: ', model_args)

    # MODIFICATION TO MODEL PARAMETERS
    new_model_args_dict = vars(model_args)
    new_model_args_dict['add_togo'] = False
    new_model_args_dict['multi_headed'] = False
    new_model_args_dict['num_classification_layers'] = 1
    new_model_args_dict['max_epochs'] = 20 # Just for dev
    new_model_args_dict['accelerator'] = 'gpu' 
    new_model_args_dict['gpus'] = 0 
    new_model_args = Namespace(**new_model_args_dict)

    # INITIALIZE MODEL
    print('New model arguments: ', new_model_args)
    model = STR2MODEL["land_cover"](new_model_args)

    last_model, trainer = train_model(model, new_model_args)

    #trainer.test(last_model) # can also pass a checkpoint to trainer.test or "best" in newer Lightnight versions
