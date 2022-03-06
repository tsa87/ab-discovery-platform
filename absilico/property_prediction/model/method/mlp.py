import numpy as np
from typing import Any, Dict

from torch import nn
import torch.nn.functional as F

# Default model config
DROP_OUT = 0.5
FC1_DIM = 128
FC2_DIM = 64


class MLP(nn.Module):
    """Simple 2-layer MLP baseline.

    Args:
        data_config (dictionary, optional): TODO
        model_config (dictionary, optional): TODO
    """

    def __init__(
        self, data_config: Dict[str, Any], model_config: Dict[str, Any] = {}
    ) -> None:
        super().__init__()

        input_dim = np.prod(data_config["input_dim"])
        output_dim = len(data_config["output_dim"])

        dropout_rate = model_config.get("drop_out", 0.5)
        fc1_dim = model_config.get("fc1_dim", 0.5)
        fc2_dim = model_config.get("fc2_dim", 0.5)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.Softmax(x)
        return x
