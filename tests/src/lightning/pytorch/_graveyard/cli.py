import sys
from typing import Any


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
    self = sys.modules[__name__]
    sys.modules["lightning.pytorch.utilities.cli"] = self


class LightningCLI:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`lightning.pytorch.utilities.cli.LightningCLI` was deprecated in v1.7.0 and is no"
            " longer supported as of v1.9.0. Please use `lightning.pytorch.cli.LightningCLI` instead"
        )


class SaveConfigCallback:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`lightning.pytorch.utilities.cli.SaveConfigCallback` was deprecated in v1.7.0 and is no"
            " longer supported as of v1.9.0. Please use `lightning.pytorch.cli.SaveConfigCallback` instead"
        )


class LightningArgumentParser:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`lightning.pytorch.utilities.cli.LightningArgumentParser` was deprecated in v1.7.0 and is no"
            " longer supported as of v1.9.0. Please use `lightning.pytorch.cli.LightningArgumentParser` instead"
        )


def instantiate_class(*_: Any, **__: Any) -> None:
    raise NotImplementedError(
        "`lightning.pytorch.utilities.cli.instantiate_class` was deprecated in v1.7.0 and is no"
        " longer supported as of v1.9.0. Please use `lightning.pytorch.cli.instantiate_class` instead"
    )


_patch_sys_modules()
