import pytorch_lightning as pl

from absilico.property_prediction.data.base import SequenceDataModule
from absilico.property_prediction.model.base import BaseLitModel
from absilico.property_prediction.model.method.mlp import MLP


data = SequenceDataModule(
    dataset_config={
        "train_csv_path": "platform/property_prediction/data/datastore/mHER/mHER_H3_test_20.csv",
        "test_csv_path": "platform/property_prediction/data/datastore/mHER/mHER_H3_train_20.csv",
    },
    transform=None,
)

# model = MLP(
#     data_config={
#         "input_dim": 26,
#         "output_dim": 2,
#     },
# )
# lit_model = BaseLitModel(
#     model=model,
# )

# trainer = pl.Trainer()
# trainer.tune(lit_model, datamodule=data)
# trainer.fit(lit_model, datamodule=data)
# trainer.test(lit_model, datamodule=data)
