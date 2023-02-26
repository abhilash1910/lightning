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
import sys
from typing import Any


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
    self = sys.modules[__name__]
    sys.modules["lightning.pytorch.profiler.advanced"] = self
    sys.modules["lightning.pytorch.profiler.base"] = self
    sys.modules["lightning.pytorch.profiler.profiler"] = self
    sys.modules["lightning.pytorch.profiler.pytorch"] = self
    sys.modules["lightning.pytorch.profiler.simple"] = self
    sys.modules["lightning.pytorch.profiler.xla"] = self


class AbstractProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`lightning.pytorch.profiler.base.AbstractProfiler` was deprecated in v1.6 and is no longer supported"
            " as of v1.9. Use `lightning.pytorch.profilers.Profiler` instead."
        )


class BaseProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.base.BaseProfiler` was deprecated in v1.6 and is no longer supported"
            " as of v1.9. Use `lightning.pytorch.profilers.Profiler` instead."
        )


class AdvancedProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.advanced.AdvancedProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.AdvancedProfiler` instead."
        )


class PassThroughProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.base.PassThroughProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.PassThroughProfiler` instead."
        )


class Profiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.profiler.Profiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.Profiler` instead."
        )


class PyTorchProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.pytorch.PyTorchProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.PyTorchProfiler` instead."
        )


class RegisterRecordFunction:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.pytorch.RegisterRecordFunction` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.pytorch.RegisterRecordFunction` instead."
        )


class ScheduleWrapper:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.pytorch.ScheduleWrapper` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.pytorch.ScheduleWrapper` instead."
        )


class SimpleProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.simple.SimpleProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.SimpleProfiler` instead."
        )


class XLAProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`lightning.pytorch.profiler.xla.XLAProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `lightning.pytorch.profilers.XLAProfiler` instead."
        )


_patch_sys_modules()
