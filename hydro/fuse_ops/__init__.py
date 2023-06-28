import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv1d, Conv2d, ConvTranspose2d
from .linear import Linear
from .pooling import MaxPool2d, AdaptiveAvgPool2d
from .dropout import Dropout2d, Dropout
from .batchnorm import BatchNorm1d, BatchNorm2d
from .sparse import Embedding
from .normalization import LayerNorm
from .activation import MultiheadAttention
from .transformer import TransformerEncoderLayer


OPS_MAP = {
    nn.Conv1d: Conv1d,
    nn.Conv2d: Conv2d,
    nn.ConvTranspose2d: ConvTranspose2d,
    nn.Linear: Linear,
    nn.MaxPool2d: MaxPool2d,
    nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
    nn.Dropout2d: Dropout2d,
    nn.BatchNorm1d: BatchNorm1d,
    nn.BatchNorm2d: BatchNorm2d,
    nn.LayerNorm: LayerNorm,
    nn.Embedding: Embedding,
    nn.TransformerEncoderLayer: TransformerEncoderLayer,
}

FUNCTION_MAP = {
    "adaptive_avg_pool2d": AdaptiveAvgPool2d,
}


UNCHANGE_OPS = [
    nn.Identity,
    nn.ReLU,
    nn.ReLU6,
    nn.Tanh,
    nn.LeakyReLU,
    nn.Dropout,
]
