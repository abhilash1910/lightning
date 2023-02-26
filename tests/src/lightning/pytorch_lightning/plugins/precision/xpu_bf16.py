# Copyright The Intel PyTorch Lightning team.
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
from functools import partial
from contextlib import contextmanager
from typing import Any, cast, Generator, List, Tuple
import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from torch import FloatTensor, Tensor
from torch.optim import Optimizer


import lightning.pytorch_lightning as pl
from lightning.lightning_fabric.accelerators.xpu import XPUAccelerator
from lightning.lightning_fabric.utilities.types import Optimizable
from lightning.pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from lightning.pytorch_lightning.utilities.exceptions import MisconfigurationException
import intel_extension_for_pytorch as ipex  # noqa: F401
import oneccl_bindings_for_pytorch  #noqa: F401

FLOAT_TYPES = (torch.FloatTensor, torch.xpu.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.xpu.HalfTensor)
dpcpp_device='xpu'

class XPUPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for TPU integration."""
    

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not XPUAccelerator.is_available():
            raise ModuleNotFoundError(str(XPUAccelerator))
        super().__init__(*args, **kwargs)
        
    def connect(
        self, model: nn.Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[nn.Module, List["Optimizer"], List[Any]]:
        """Converts the model to double precision and wraps it in a ``LightningDoublePrecisionModule`` to convert
        incoming floating point data to double (``torch.float64``) precision.

        Does not alter `optimizers` or `lr_schedulers`.
        """
        model = model.to(dpcpp_device)
        model = cast(pl.LightningModule, model)

        return super().connect(model, optimizers, lr_schedulers)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type.

        See: :meth:`torch.set_default_tensor_type`
        """
        torch.set_default_tensor_type(torch.xpu.DoubleTensor)
        yield
        torch.set_default_tensor_type(torch.xpu.FloatTensor)



class XPUBf16PrecisionPlugin(PrecisionPlugin):
    """Precision plugin for TPU integration."""
    
    precision: str = "bf16"
    

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not XPUAccelerator.is_available():
            raise ModuleNotFoundError(str(XPUAccelerator))
        super().__init__(*args, **kwargs)
        
    def connect(
        self, model: nn.Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[nn.Module, List["Optimizer"], List[Any]]:
        """Converts the model to double precision and wraps it in a ``LightningDoublePrecisionModule`` to convert
        incoming floating point data to double (``torch.float64``) precision.

        Does not alter `optimizers` or `lr_schedulers`.
        """
        model = model.to(dpcpp_device,dtype=torch.bfloat16)
        model = cast(pl.LightningModule, model)
        #model = (model)

        return super().connect(model, optimizers, lr_schedulers)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type.

        See: :meth:`torch.set_default_tensor_type`
        """
        torch.set_default_tensor_type(torch.xpu.DoubleTensor)
        yield
        torch.set_default_tensor_type(torch.xpu.FloatTensor)


