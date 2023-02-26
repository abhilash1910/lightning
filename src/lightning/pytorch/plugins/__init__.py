from typing import Union

from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO, XLACheckpointIO
from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO
from lightning.pytorch.plugins.io.hpu_plugin import HPUCheckpointIO
from lightning.pytorch.plugins.layer_sync import LayerSync, NativeSyncBatchNorm
from lightning.pytorch.plugins.precision.apex_amp import ApexMixedPrecisionPlugin
from lightning.pytorch.plugins.precision.colossalai import ColossalAIPrecisionPlugin
from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin
from lightning.pytorch.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from lightning.pytorch.plugins.precision.fully_sharded_native_amp import FullyShardedNativeMixedPrecisionPlugin
from lightning.pytorch.plugins.precision.hpu import HPUPrecisionPlugin
from lightning.pytorch.plugins.precision.ipu import IPUPrecisionPlugin
from lightning.pytorch.plugins.precision.native_amp import MixedPrecisionPlugin, NativeMixedPrecisionPlugin
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
from lightning.pytorch.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from lightning.pytorch.plugins.precision.tpu import TPUPrecisionPlugin
from lightning.pytorch.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin

PLUGIN = Union[PrecisionPlugin, ClusterEnvironment, CheckpointIO, LayerSync]
PLUGIN_INPUT = Union[PLUGIN, str]

__all__ = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "HPUCheckpointIO",
    "ApexMixedPrecisionPlugin",
    "ColossalAIPrecisionPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "IPUPrecisionPlugin",
    "HPUPrecisionPlugin",
    "NativeMixedPrecisionPlugin",
    "MixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "FullyShardedNativeMixedPrecisionPlugin",
    "FullyShardedNativeNativeMixedPrecisionPlugin",
    "TPUPrecisionPlugin",
    "TPUBf16PrecisionPlugin",
    "LayerSync",
    "NativeSyncBatchNorm",
]
