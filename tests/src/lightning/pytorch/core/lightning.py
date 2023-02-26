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
from typing import Any

from lightning.pytorch.core.module import LightningModule as NewLightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_deprecation


class LightningModule(NewLightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "lightning.pytorch.core.lightning.LightningModule has been deprecated in v1.7"
            " and will be removed in v1.9."
            " Use the equivalent class from the lightning.pytorch.core.module.LightningModule class instead."
        )
        super().__init__(*args, **kwargs)
