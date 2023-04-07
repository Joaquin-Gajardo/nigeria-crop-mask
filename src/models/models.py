import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.utils import set_seed
from .model_bases import STR2BASE
from .data import LandTypeClassificationDataset, GeowikiDataset
from .utils import tif_to_np, preds_to_xr

from typing import cast, Callable, Tuple, Dict, Any, Type, Optional, List, Union


class LandCoverMapper(pl.LightningModule):
    r"""
    An LSTM based model to predict the presence of cropland
    inside a pixel.

    hparams
    --------
    The default values for these parameters are set in add_model_specific_args

    :param hparams.data_folder: The path to the data. Default (assumes the model
        is being run from the scripts directory) = "../data"
    :param hparams.model_base: The model base to use. Currently, only an LSTM
        is implemented. Default = "lstm"
    :param hparams.hidden_vector_size: The size of the hidden vector. Default = 64
    :param hparams.learning_rate: The learning rate. Default = 0.001
    :param hparams.batch_size: The batch size. Default = 64
    :param hparams.probability_threshold: The probability threshold to use to label GeoWiki
        instances as crop / not_crop (since the GeoWiki labels are a mean crop probability, as
        assigned by several labellers). In addition, this is the threshold used when calculating
        metrics which require binary predictions, such as accuracy score. Default = 0.5
    :param hparams.num_classification_layers: The number of classification layers to add after
        the base. Default = 1
    :param hparams.classification_dropout: Dropout ratio to apply on the hidden vector before
        it is passed to the classification layer(s). Default = 0
    :param hparams.alpha: The weight to use when adding the global and local losses. This parameter
        is only used if hparams.multi_headed is True. Default = 10
    :param hparams.add_togo: Whether or not to use the hand labelled dataset to train the model.
        Default = True
    :param hparams.add_geowiki: Whether or not to use the GeoWiki dataset to train the model.
        Default = True
    :param hparams.remove_b1_b10: Whether or not to remove the B1 and B10 bands. Default = True
    :param hparams.multi_headed: Whether or not to add a local head, to classify instances within
        Togo. If False, the same classification layer will be used to classify
        all pixels. Default = True
    :param hparams.weighted_loss_fn: Whether or not to use weighted loss function (by class weights). Default = False
    :param hparams.geowiki_subset: List of countries to use from geowiki. Choices=["nigeria", "neighbours1", "neighbours2", "world"]. Default = "world".
    :param hparams.add_nigeria: Whether or not to use the Nigeria dataset to train the model.
        Default = True
    """

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        set_seed() # NOTE: will probably have to unset if I want to do several runs
        hparams = Namespace(**hparams) if not isinstance(hparams, Namespace) else hparams
        self.hparams = hparams
        
        # Dataset
        self.data_folder = Path(hparams.data_folder)

        if self.hparams.add_geowiki:
            if hparams.geowiki_subset == 'nigeria':
                countries_subset = ['Nigeria']
            elif hparams.geowiki_subset == 'neighbours1':
                countries_subset = ['Ghana', 'Togo', 'Nigeria', 'Cameroon', 'Benin']
            elif hparams.geowiki_subset == 'neighbours2':
                countries_subset = ['Nigeria', 'Benin', 'Niger', 'Chad', 'Cameroon']
            else: # world
                countries_subset = None
            geowiki_dataset = GeowikiDataset(
                data_folder=self.data_folder,
                countries_subset=countries_subset,
                countries_to_weight=['Nigeria'],
                remove_b1_b10=self.hparams.remove_b1_b10,
                crop_probability_threshold=self.hparams.probability_threshold,
            )

            geowiki_dataset_splits = geowiki_dataset.train_val_split(geowiki_dataset)
            self.geowiki_train_dataset = geowiki_dataset_splits[0]
            self.geowiki_val_dataset = geowiki_dataset_splits[1]

        dataset = self.get_dataset(subset="training")

        self.input_size = dataset.num_input_features
        self.num_outputs = dataset.num_output_classes

        # we save the normalizing dict because we calculate weighted
        # normalization values based on the datasets we combine. --> from adjust_normalizing_dict function
        # The number of instances per dataset (and therefore the weights) can
        # vary between the train / test / val sets - this ensures the normalizing
        # dict stays constant between them
        ###  See NOTE on test_dataloader for explanation ###
        self.normalizing_dict = dataset.normalizing_dict # -> we'll use it for test set

        if self.hparams.weighted_loss_fn:
            val_dataset = self.get_dataset(subset="validation")
            self.global_class_weights, self.local_class_weights = self.get_class_weights([dataset, val_dataset])
            print('Global class weights:', self.global_class_weights)
            print('Local class weights:', self.local_class_weights)

        # Model layers       
        self.model_base_name = hparams.model_base

        self.base = STR2BASE[hparams.model_base](
            input_size=self.input_size, hparams=self.hparams
        )

        global_classification_layers: List[nn.Module] = []
        for i in range(hparams.num_classification_layers):
            global_classification_layers.append(
                nn.Linear(
                    in_features=hparams.hidden_vector_size,
                    out_features=self.num_outputs
                    if i == (hparams.num_classification_layers - 1)
                    else hparams.hidden_vector_size,
                )
            )
            if i < (hparams.num_classification_layers - 1):
                global_classification_layers.append(nn.ReLU())

        self.global_classifier = nn.Sequential(*global_classification_layers)

        if self.hparams.multi_headed:

            local_classification_layers: List[nn.Module] = []
            for i in range(hparams.num_classification_layers):
                local_classification_layers.append(
                    nn.Linear(
                        in_features=hparams.hidden_vector_size,
                        out_features=self.num_outputs
                        if i == (hparams.num_classification_layers - 1)
                        else hparams.hidden_vector_size,
                    )
                )
                if i < (hparams.num_classification_layers - 1):
                    local_classification_layers.append(nn.ReLU())

            self.local_classifier = nn.Sequential(*local_classification_layers)

        self.loss_function: Callable = F.binary_cross_entropy

        print('Number of model parameters:', self.num_trainable_parameters)

    @property
    def num_trainable_parameters(self): 
        return sum(param.numel() for param in self.parameters() if param.requires_grad_)

    def get_class_weights(self, datasets: List):
        global_labels = []
        local_labels = []
        for dataset in datasets:
            for _, label, weight in dataset:
                # If we have only one head (not multiheaded) everything should
                # be a global label because eveything goes to the global head
                if not self.hparams.multi_headed or weight.item() == 0: 
                    global_labels.append(label.reshape(1))
                else:
                    local_labels.append(label.reshape(1))

        print('Number of global labels:', len(global_labels))
        print('Number of local labels:', len(local_labels))
        
        # For case of multiheaded without any weighted sample, this should never happen in practice though
        # as it doesn't make sense to use multiheaded if if not using data out of the target country
        global_class_weights = None 
        if len(global_labels) != 0: 
            global_labels = torch.cat(global_labels)
            global_class_dist = torch.bincount(global_labels.to(torch.int))
            global_class_weights = global_class_dist.sum() / global_class_dist#.to(torch.float32) # to float so that result is not cast to int

        # To handle opposite case of multiheaded without any local labels found
        local_class_weights = None
        if self.hparams.multi_headed:
            assert len(local_labels) != 0, 'If multiheaded all labels should be global so they are use in the global head'
            local_labels = torch.cat(local_labels)
            local_class_dist = torch.bincount(local_labels.to(torch.int))
            local_class_weights = local_class_dist.sum() / local_class_dist#.to(torch.float32) # to float so that result is not cast to int

        return global_class_weights, local_class_weights

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        base = self.base(x)
        x_global = self.global_classifier(base)

        if self.num_outputs == 1:
            x_global = torch.sigmoid(x_global)

        if self.hparams.multi_headed:
            x_local = self.local_classifier(base)
            if self.num_outputs == 1:
                x_local = torch.sigmoid(x_local)
            return x_global, x_local

        else:
            return x_global

    def get_dataset(
        self, subset: str, normalizing_dict: Optional[Dict] = None, evaluating: bool = False
    ) -> LandTypeClassificationDataset:
        geowiki_set = None
        if self.hparams.add_geowiki:
            if subset == 'training':
                geowiki_set = self.geowiki_train_dataset
            elif subset == 'validation':
                geowiki_set = self.geowiki_val_dataset

        return LandTypeClassificationDataset(
            data_folder=self.data_folder,
            subset=subset,
            crop_probability_threshold=self.hparams.probability_threshold,
            include_geowiki=self.hparams.add_geowiki,
            include_togo=self.hparams.add_togo,
            include_nigeria=self.hparams.add_nigeria,
            normalizing_dict=normalizing_dict,
            remove_b1_b10=self.hparams.remove_b1_b10,
            evaluating=evaluating,
            geowiki_set=geowiki_set,
        )

    def train_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="training"),
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count()  
        )                                           

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="validation", normalizing_dict=self.normalizing_dict),
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count()  
        )

    def test_dataloader(self):
        '''
        NOTE about normalizing dict:
        Same normalizing dict as in train and val dataset is passed. It's calculated from all train and val samples used.
        Additionally, if different datasets are used (geowiki, togo) the normalization values are averaged in terms of the ammount of samples (see adjust_normalizing_dict).
        '''  
        return DataLoader(
            self.get_dataset(subset="validation", normalizing_dict=self.normalizing_dict, evaluating=True),
            #self.get_dataset(subset="testing", normalizing_dict=self.normalizing_dict, evaluating=True),
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count()  
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        loss = self._split_preds_and_get_loss(
            batch, add_preds=False, loss_label="loss", log_loss=True
        )
        self.logger.log_metrics({'train loss': loss["loss"]})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="val_loss", log_loss=False
        )
        self.logger.log_metrics({'val loss': loss["val_loss"]})
        return loss

    def test_step(self, batch, batch_idx):
        loss =  self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="test_loss", log_loss=False
        )
        #self.logger.log_metrics({'test loss': loss["test_loss"]})
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss}
        tensorboard_logs.update(self.get_interpretable_metrics(outputs, prefix="val_"))

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()

        output_dict = {"test_loss": avg_loss}
        output_dict.update(self.get_interpretable_metrics(outputs, prefix="test_"))

        return {"progress_bar": output_dict}

    def save_validation_predictions(self) -> None:
        """
        This can be useful in combination with src.utils.plot_roc_curve
        to find an appropriate threshold.
        """
        save_dir = (
            Path(self.hparams.data_folder) / self.__class__.__name__ / "validation"
        )
        save_dir.mkdir(exist_ok=True, parents=True)

        val_dl = self.val_dataloader()

        outputs: List[Dict] = []
        for idx, batch in enumerate(val_dl):

            with torch.no_grad():
                outputs.append(self.validation_step(batch, idx))

        all_preds = (torch.cat([x["pred"] for x in outputs]).detach().cpu().numpy(),)
        all_labels = (torch.cat([x["label"] for x in outputs]).detach().cpu().numpy(),)

        np.save(save_dir / "all_preds.npy", all_preds)
        np.save(save_dir / "all_labels.npy", all_labels)

        if self.hparams.multi_headed:
            r_preds = (
                torch.cat([x["Togo_pred"] for x in outputs]).detach().cpu().numpy()
            )
            r_labels = (
                torch.cat([x["Togo_label"] for x in outputs]).detach().cpu().numpy()
            )

            np.save(save_dir / "Togo_preds.npy", r_preds)
            np.save(save_dir / "Togo_labels.npy", r_labels)

    def predict(
        self,
        path_to_file: Path,
        batch_size: int = 64,
        add_ndvi: bool = True,
        nan_fill: float = 0,
        days_per_timestep: int = 30,
        local_head: bool = True,
        use_gpu: bool= False,
    ) -> xr.Dataset:

        # check if a GPU is available, and if it is
        # move the model onto the GPU
        device: Optional[torch.device] = None
        if use_gpu:
            use_cuda = torch.cuda.is_available()
            if not use_cuda:
                print("No GPU - not using one")
            device = torch.device("cuda" if use_cuda else "cpu")
            self.to(device)

        self.eval()

        input_data = tif_to_np(
            path_to_file,
            add_ndvi=add_ndvi,
            nan=nan_fill,
            normalizing_dict=self.normalizing_dict,
            days_per_timestep=days_per_timestep,
        )

        dataset = self.get_dataset(subset="training")

        predictions: List[np.ndarray] = []
        cur_i = 0

        pbar = tqdm(total=input_data.x.shape[0] - 1)
        while cur_i < (input_data.x.shape[0] - 1):
            batch_x = torch.from_numpy(
                dataset.remove_bands(input_data.x[cur_i : cur_i + batch_size])
            ).float()

            if use_gpu and (device is not None):
                batch_x = batch_x.to(device)

            with torch.no_grad():
                batch_preds = self.forward(batch_x)

                if self.hparams.multi_headed:
                    global_preds, local_preds = batch_preds

                    if local_head:
                        batch_preds = local_preds
                    else:
                        batch_preds = global_preds

                if self.num_outputs > 1:
                    batch_preds = F.softmax(cast(torch.Tensor, batch_preds), dim=-1)

                # back to the CPU, if necessary
                batch_preds = batch_preds.cpu()

            predictions.append(cast(torch.Tensor, batch_preds).numpy())
            cur_i += batch_size
            pbar.update(batch_size)

        all_preds = np.concatenate(predictions, axis=0)
        if len(all_preds.shape) == 1:
            all_preds = np.expand_dims(all_preds, axis=-1)

        return preds_to_xr(
            all_preds, lats=input_data.lat, lons=input_data.lon, feature_labels=None
        )

    def get_interpretable_metrics(self, outputs, prefix: str) -> Dict:

        output_dict = {}

        # we want to calculate some more interpretable losses - accuracy,
        # and auc roc
        output_dict.update(
            self.single_output_metrics(
                torch.cat([x["pred"] for x in outputs]).detach().cpu().numpy(),
                torch.cat([x["label"] for x in outputs]).detach().cpu().numpy(),
                prefix=prefix,
            )
        )

        if self.hparams.multi_headed:
            output_dict.update(
                self.single_output_metrics(
                    torch.cat([x["Togo_pred"] for x in outputs]).detach().cpu().numpy(),
                    torch.cat([x["Togo_label"] for x in outputs])
                    .detach()
                    .cpu()
                    .numpy(),
                    prefix=f"{prefix}Togo_",
                )
            )
        return output_dict

    def single_output_metrics(
        self, preds: np.ndarray, labels: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:

        if len(preds) == 0:
            # sometimes this happens in the warmup. Or when using multiheaded and there are no global labels
            return {}

        output_dict: Dict[str, float] = {}
        if not (labels == labels[0]).all():
            # This can happen when lightning does its warm up on a subset of the
            # validation data
            output_dict[f"{prefix}roc_auc_score"] = roc_auc_score(labels, preds)

        preds = (preds > self.hparams.probability_threshold).astype(int)

        output_dict[f"{prefix}precision_score"] = precision_score(labels, preds, zero_division=0)
        output_dict[f"{prefix}recall_score"] = recall_score(labels, preds)
        output_dict[f"{prefix}f1_score"] = f1_score(labels, preds)
        output_dict[f"{prefix}accuracy"] = accuracy_score(labels, preds)
        print('confusion matrix:', confusion_matrix(labels, preds))
        return output_dict

    def _split_preds_and_get_loss(
        self, batch, add_preds: bool, loss_label: str, log_loss: bool
    ) -> Dict:

        x, label, is_togo = batch

        preds_dict: Dict = {}
        if self.hparams.multi_headed:
            global_preds, local_preds = self.forward(x)
            global_preds = global_preds[is_togo == 0]
            global_labels = label[is_togo == 0]

            local_preds = local_preds[is_togo == 1]
            local_labels = label[is_togo == 1]

            loss = 0
            if local_preds.shape[0] > 0:
                if not self.hparams.weighted_loss_fn:
                    local_loss = self.loss_function(local_preds.squeeze(-1), local_labels)
                else:
                    local_loss = self.weighted_loss_function(local_preds, local_labels, self.local_class_weights) 
                loss += local_loss

            if global_preds.shape[0] > 0:
                if not self.hparams.weighted_loss_fn:
                    global_loss = self.loss_function(global_preds.squeeze(-1), global_labels)
                else:
                    global_loss = self.weighted_loss_function(global_preds, global_labels, self.global_class_weights) 

                num_local_labels = local_preds.shape[0]
                if num_local_labels == 0:
                    alpha = 1
                else:
                    ratio = global_preds.shape[0] / num_local_labels
                    alpha = ratio / self.hparams.alpha

                loss += alpha * global_loss
            if add_preds:
                preds_dict.update(
                    {
                        "pred": global_preds,
                        "label": global_labels,
                        "Togo_pred": local_preds,
                        "Togo_label": local_labels,
                    }
                )
        else:
            # "is_togo" (weights) will be ignored
            preds = self.forward(x)

            if not self.hparams.weighted_loss_fn:
                loss = self.loss_function(input=cast(torch.Tensor, preds).squeeze(-1), target=label)
            else:
                loss = self.weighted_loss_function(preds, label, self.global_class_weights) # if not multiheaded there are only global class weights

            if add_preds:
                preds_dict.update({"pred": preds, "label": label})

        output_dict: Dict = {loss_label: loss}
        if log_loss:
            output_dict["log"] = {loss_label: loss}
        output_dict.update(preds_dict)
        return output_dict

    def weighted_loss_function(self, preds: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
        assert class_weights.shape[0] == 2, 'Labels are supossed to be binary so class weights should have two elements.'
        batch_weights = torch.ones(labels.size(0)) #TODO: add to device when using GPU
        for i in [0, 1]:
            batch_weights = torch.where(labels == float(i), class_weights[i].to(torch.float32), batch_weights)
        loss = (batch_weights * self.loss_function(input=cast(torch.Tensor, preds).squeeze(-1), target=labels, reduction='none')).mean()
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # by default, this means no additional weighting will be given
        # if a region is passed, then we will assign a weighting of 10
        # (this happens in the dataloader, and empirically seems to work well. If
        # we do more experimenting with the hparams it might make sense to make it
        # modifiable here).
        parser_args: Dict[str, Tuple[Type, Any]] = {
            "--data_folder": (
                str,
                # this allows the model to be run from
                # anywhere on the machine
                str(Path("../data").absolute()),
            ),  # assumes this is being run from "scripts"
            "--model_base": (str, "lstm"),
            "--hidden_vector_size": (int, 64),
            "--learning_rate": (float, 0.001),
            "--batch_size": (int, 64),
            "--probability_threshold": (float, 0.5),
            "--num_classification_layers": (int, 2),
            "--alpha": (float, 10),
        }

        for key, val in parser_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        parser.add_argument("--add_togo", dest="add_togo", action="store_true")
        parser.add_argument("--exclude_togo", dest="add_togo", action="store_false")
        parser.set_defaults(add_togo=False)

        parser.add_argument("--add_geowiki", dest="add_geowiki", action="store_true")
        parser.add_argument(
            "--exclude_geowiki", dest="add_geowiki", action="store_false"
        )
        parser.set_defaults(add_geowiki=True)

        parser.add_argument("--add_nigeria", dest="add_nigeria", action="store_true")
        parser.add_argument("--exclude_nigeria", dest="add_nigeria", action="store_false")
        parser.set_defaults(add_nigeria=True)

        parser.add_argument("--geowiki_subset", default="world", choices=["nigeria", "neighbours1", "neighbours2", "world"], help="It will be ignored if geowiki was excluded.")
        
        parser.add_argument(
            "--remove_b1_b10", dest="remove_b1_b10", action="store_true"
        )
        parser.add_argument("--keep_b1_b10", dest="remove_b1_b10", action="store_false")
        parser.set_defaults(remove_b1_b10=True)

        parser.add_argument("--multi_headed", dest="multi_headed", action="store_true")
        parser.add_argument(
            "--not_multi_headed", dest="multi_headed", action="store_false"
        )
        parser.set_defaults(multi_headed=False)

        temp_args = parser.parse_known_args()[0]
        return STR2BASE[temp_args.model_base].add_base_specific_arguments(parser)
