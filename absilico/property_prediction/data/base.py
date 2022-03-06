import pandas as pd
import pytorch_lightning as pl
from absilico.property_prediction.data.utils import BaseDataset, split_dataset
from typing import Any, Callable, Dict, Optional, Union

from torch.utils.data import ConcatDataset, DataLoader

BATCH_SIZE = 64
NUM_WORKERS = 0
TRAIN_FRAC = 0.8
SEED = 0


class SequenceDataModule(pl.LightningDataModule):
    """
    A Pytorch dataset that contains antibody sequence strings and assay labels.

    Args:
        sequence_col (int): Column index of antibody sequence string in csv.
        label_col (int): Column index of assay label in csv.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    """

    def __init__(
        self,
        dataset_config: Dict[str, Any],
        transform: Callable = None,
    ):
        self.train_csv_path = dataset_config["train_csv_path"]
        self.test_csv_path = dataset_config["test_csv_path"]

        self.batch_size = dataset_config.get("batch_size", BATCH_SIZE)
        self.num_workers = dataset_config.get("num_workers", NUM_WORKERS)
        self.on_gpu = dataset_config.get("on_gpu", False)
        self.train_frac = dataset_config.get("train_frac", TRAIN_FRAC)
        self.seed = dataset_config.get("seed", SEED)

        self.transform = transform

        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train = pd.read_csv(self.train_csv_path)
            train_X = train[self.sequence_col]
            train_Y = train[self.label_col]
            data_trainval = BaseDataset(
                train_X, train_Y, transform=self.transform
            )
            self.data_train, self.data_val = split_dataset(
                data_trainval, self.train_frac, self.seed
            )

        if stage == "test" or stage is None:
            test = pd.read_csv(self.test_csv_path)
            test_X = test[self.sequence_col]
            test_Y = test[self.label_col]
            self.data_test = BaseDataset(
                test_X, test_Y, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
