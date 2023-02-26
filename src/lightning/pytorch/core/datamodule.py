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
"""LightningDataModule for loading DataLoaders with ease."""
import inspect
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, IO, List, Mapping, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import Self

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.hooks import DataHooks
from lightning.pytorch.core.mixins import HyperparametersMixin
from lightning.pytorch.core.saving import _load_from_checkpoint
from lightning.pytorch.utilities.argparse import (
    add_argparse_args,
    from_argparse_args,
    get_init_arguments_and_types,
    parse_argparser,
)
from lightning.pytorch.utilities.types import _ADD_ARGPARSE_RETURN, EVAL_DATALOADERS, TRAIN_DATALOADERS


class LightningDataModule(DataHooks, HyperparametersMixin):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms. The main
    advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
            def teardown(self):
                # clean up after fit or test
                # called on every process in DDP
    """

    name: Optional[str] = None
    CHECKPOINT_HYPER_PARAMS_KEY = "datamodule_hyper_parameters"
    CHECKPOINT_HYPER_PARAMS_NAME = "datamodule_hparams_name"
    CHECKPOINT_HYPER_PARAMS_TYPE = "datamodule_hparams_type"

    def __init__(self) -> None:
        super().__init__()
        # Pointer to the trainer object
        self.trainer: Optional["pl.Trainer"] = None

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs: Any) -> _ADD_ARGPARSE_RETURN:
        """Extends existing argparse by default `LightningDataModule` attributes.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = LightningDataModule.add_argparse_args(parser)
        """
        return add_argparse_args(cls, parent_parser, **kwargs)

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs: Any
    ) -> Union["pl.LightningDataModule", "pl.Trainer"]:
        """Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`~lightning.pytorch.core.datamodule.LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid DataModule arguments.

        Example::

            module = LightningDataModule.from_argparse_args(args)
        """
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return parse_argparser(cls, arg_parser)

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""Scans the DataModule signature and returns argument names, types and default values.

        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).
        """
        return get_init_arguments_and_types(cls)

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **datamodule_kwargs: Any,
    ) -> "LightningDataModule":
        r"""
        Create an instance from torch.utils.data.Dataset.

        Args:
            train_dataset: Optional dataset to be used for train_dataloader()
            val_dataset: Optional dataset or list of Dataset to be used for val_dataloader()
            test_dataset: Optional dataset or list of Dataset to be used for test_dataloader()
            predict_dataset: Optional dataset or list of Dataset to be used for predict_dataloader()
            batch_size: Batch size to use for each dataloader. Default is 1. This parameter gets forwarded to the
                ``__init__`` if the datamodule has such a name defined in its signature.
            num_workers: Number of subprocesses to use for data loading. 0 means that the
                data will be loaded in the main process. Number of CPUs available. This parameter gets forwarded to the
                ``__init__`` if the datamodule has such a name defined in its signature.
            **datamodule_kwargs: Additional parameters that get passed down to the datamodule's ``__init__``.
        """

        def dataloader(ds: Dataset, shuffle: bool = False) -> DataLoader:
            shuffle &= not isinstance(ds, IterableDataset)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

        def train_dataloader() -> TRAIN_DATALOADERS:
            assert train_dataset

            if isinstance(train_dataset, Mapping):
                return {key: dataloader(ds, shuffle=True) for key, ds in train_dataset.items()}
            if isinstance(train_dataset, Sequence):
                return [dataloader(ds, shuffle=True) for ds in train_dataset]
            return dataloader(train_dataset, shuffle=True)

        def val_dataloader() -> EVAL_DATALOADERS:
            assert val_dataset

            if isinstance(val_dataset, Sequence):
                return [dataloader(ds) for ds in val_dataset]
            return dataloader(val_dataset)

        def test_dataloader() -> EVAL_DATALOADERS:
            assert test_dataset

            if isinstance(test_dataset, Sequence):
                return [dataloader(ds) for ds in test_dataset]
            return dataloader(test_dataset)

        def predict_dataloader() -> EVAL_DATALOADERS:
            assert predict_dataset

            if isinstance(predict_dataset, Sequence):
                return [dataloader(ds) for ds in predict_dataset]
            return dataloader(predict_dataset)

        candidate_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
        accepted_params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in accepted_params.values())
        if accepts_kwargs:
            special_kwargs = candidate_kwargs
        else:
            accepted_param_names = set(accepted_params)
            accepted_param_names.discard("self")
            special_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_param_names}

        datamodule = cls(**datamodule_kwargs, **special_kwargs)
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader  # type: ignore[assignment]
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader  # type: ignore[assignment]
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader  # type: ignore[assignment]
        if predict_dataset is not None:
            datamodule.predict_dataloader = predict_dataloader  # type: ignore[assignment]
        return datamodule

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.
        """
        return dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.
        """
        pass

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        hparams_file: Optional[_PATH] = None,
        **kwargs: Any,
    ) -> Self:  # type: ignore[valid-type]
        r"""
        Primary way of loading a datamodule from a checkpoint. When Lightning saves a checkpoint
        it stores the arguments passed to ``__init__``  in the checkpoint under ``"datamodule_hyper_parameters"``.

        Any arguments specified through \*\*kwargs will override args stored in ``"datamodule_hyper_parameters"``.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
            hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                as in this example::

                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningDataModule` for use.

                If your datamodule's ``hparams`` argument is :class:`~argparse.Namespace`
                and ``.yaml`` file has hierarchical structure, you need to refactor your datamodule to treat
                ``hparams`` as :class:`~dict`.
            \**kwargs: Any extra keyword args needed to init the datamodule. Can also be used to override saved
                hyperparameter values.

        Return:
            :class:`LightningDataModule` instance with loaded weights and hyperparameters (if available).

        Note:
            ``load_from_checkpoint`` is a **class** method. You should use your :class:`LightningDataModule`
            **class** to call it instead of the :class:`LightningDataModule` instance.

        Example::

            # load weights without mapping ...
            datamodule = MyLightningDataModule.load_from_checkpoint('path/to/checkpoint.ckpt')

            # or load weights and hyperparameters from separate files.
            datamodule = MyLightningDataModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                hparams_file='/path/to/hparams_file.yaml'
            )

            # override some of the params with new values
            datamodule = MyLightningDataModule.load_from_checkpoint(
                PATH,
                batch_size=32,
                num_workers=10,
            )

        """
        return _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location=None,
            hparams_file=hparams_file,
            strict=None,
            **kwargs,
        )
