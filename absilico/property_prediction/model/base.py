from typing import Any, Dict

import torch
import pytorch_lightning as pl


OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"


class BaseLitModel(pl.LightningModule):
    """Simple 2-layer MLP baseline.

    Args:
        model (): TODO
        model_config (dictionary, optional): TODO
    """

    def __init__(
        self, model: torch.nn.Modules, training_config: Dict[str, Any] = {}
    ) -> None:
        super().__init__()
        self.model = model

        optimizer = training_config.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = training_config.get("lr", LR)

        loss = training_config.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
