"""Lightning datamodule for crystal CSV splits."""

import os
import logging
from typing import Optional, Sequence

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from src.data.components.crystal_dataset import CrystalDataset
log = logging.getLogger(__name__)


class CSVCrystalDataModule(LightningDataModule):
    """Load crystal train/val/test CSV files with a CIF column.

    Expected layout:
        root/
          train.csv
          val.csv
          test.csv

    Each row must contain a ``cif`` column. A ``material_id`` column and
    optional property columns are passed through by ``CrystalDataset``.
    """

    def __init__(
        self,
        root: str,
        prop: str = "formation_energy_per_atom",
        train_file: str = "train.csv",
        val_file: str = "val.csv",
        test_file: str = "test.csv",
        processed_dir: Optional[str] = None,
        niggli: bool = True,
        primitive: bool = False,
        graph_method: str = "crystalnn",
        tolerance: float = 0.1,
        use_space_group: bool = False,
        preprocess_workers: int = 16,
        batch_size: Optional[dict] = None,
        num_workers: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def _make_dataset(self, split: str, filename: str) -> CrystalDataset:
        root = self.hparams.root
        processed_dir = self.hparams.processed_dir or os.path.join(root, "processed_crystals")
        os.makedirs(processed_dir, exist_ok=True)
        return CrystalDataset(
            name=f"{os.path.basename(root)}_{split}",
            path=os.path.join(root, filename),
            save_path=os.path.join(processed_dir, f"{split}.pt"),
            prop=self.hparams.prop,
            niggli=self.hparams.niggli,
            primitive=self.hparams.primitive,
            graph_method=self.hparams.graph_method,
            tolerance=self.hparams.tolerance,
            use_space_group=self.hparams.use_space_group,
            preprocess_workers=self.hparams.preprocess_workers,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage in ["fit", "validate"]:
            self.train_dataset = self._make_dataset("train", self.hparams.train_file)
            self.val_dataset = self._make_dataset("val", self.hparams.val_file)
            log.info(
                f"CSV crystal train dataset: {len(self.train_dataset)} samples; "
                f"val dataset: {len(self.val_dataset)} samples"
            )

        if stage is None or stage in ["test", "predict"]:
            self.test_dataset = self._make_dataset("test", self.hparams.test_file)
            log.info(f"CSV crystal test dataset: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset=self.val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                pin_memory=False,
                shuffle=False,
            )
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                pin_memory=False,
                shuffle=False,
            )
        ]
