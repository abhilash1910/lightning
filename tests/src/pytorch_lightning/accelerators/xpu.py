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
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union

import torch

import pytorch_lightning as pl
#from deepspeed.accelerator.real_accelerator import get_accelerator
import intel_extension_for_pytorch as ipex  # noqa: F401
import oneccl_bindings_for_pytorch  #noqa: F401
from lightning_fabric.utilities.device_parser import _parse_xpu_ids
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class XPUAccelerator(Accelerator):
    """Accelerator for XPU devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        if device.type != "xpu":
            raise MisconfigurationException(f"Device should be XPU, got {device} instead")
        #_check_cuda_matmul_precision(device)
        #get_accelerator().set_device(device)
        torch.xpu.set_device(device)

    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        self.set_xpu_flags(trainer.local_rank)
        # clear cache before training
        #get_accelerator().empty_cache()
        torch.xpu.empty_cache()

    @staticmethod
    def set_xpu_flags(local_rank: int) -> None:
        # set the correct cuda visible devices (using pci order)
        os.environ["XPU_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_xpu_ids = ",".join(str(x) for x in range(num_xpu_devices()))
        devices = os.getenv("XPU_VISIBLE_DEVICES", all_xpu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - XPU_VISIBLE_DEVICES: [{devices}]")

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If nvidia-smi installation not found
        """
        #return get_accelerator().memory_stats(device)
        return torch.xpu.memory_stats(device)
        
    def teardown(self) -> None:
        # clean up memory
        #get_accelerator().empty_cache()
        torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        return _parse_xpu_ids(devices, include_cuda=False)

    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_xpu_devices()

    @staticmethod
    def is_available() -> bool:
        return num_xpu_devices() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )


def num_xpu_devices() -> int:
    """Returns the number of available CUDA devices.

    Unlike :func:`torch.cuda.device_count`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.
    """
    
    # Implementation copied from upstream: https://github.com/pytorch/pytorch/pull/84879
    # TODO: Remove once minimum supported PyTorch version is 1.13
    # nvml_count = _device_count_nvml()
    #return get_accelerator().device_count() #if nvml_count < 0 else nvml_count
    return torch.xpu.device_count()


def _get_xpu_id(device_id: int) -> str:
    """Get the unmasked real GPU IDs."""
    # All devices if `CUDA_VISIBLE_DEVICES` unset
    default = ",".join(str(i) for i in range(num_xpu_devices()))
    xpu_visible_devices = os.getenv("XPU_VISIBLE_DEVICES", default=default).split(",")
    return xpu_visible_devices[device_id].strip()
