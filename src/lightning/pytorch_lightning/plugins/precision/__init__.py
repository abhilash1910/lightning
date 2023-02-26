# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from lightning.pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.colossalai import ColossalAIPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.fully_sharded_native_amp import FullyShardedNativeMixedPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.hpu import HPUPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.ipu import IPUPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.native_amp import MixedPrecisionPlugin, NativeMixedPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.tpu import TPUPrecisionPlugin
from lightning.pytorch_lightning.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin

__all__ = [
    "ApexMixedPrecisionPlugin",
    "ColossalAIPrecisionPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "FullyShardedNativeNativeMixedPrecisionPlugin",
    "FullyShardedNativeMixedPrecisionPlugin",
    "HPUPrecisionPlugin",
    "IPUPrecisionPlugin",
    "NativeMixedPrecisionPlugin",
    "MixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "TPUPrecisionPlugin",
    "TPUBf16PrecisionPlugin",
]
