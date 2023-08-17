from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from nltk.translate.bleu_score import corpus_bleu


class OutputLogger(pl.Callback):
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pl_module.log(
            "train_loss",
            outputs["loss"],
            prog_bar=True
        )

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        self.valid_loss = list()
        self.references = list()
        self.hypotheses = list()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        trainer.current_epoch
        self.valid_loss.append(outputs["loss"])
        self.references.extend(outputs["references"])
        self.hypotheses.extend(outputs["hypotheses"])

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        bleu4 = corpus_bleu(self.references, self.hypotheses)

        pl_module.log("Bleu4", float(bleu4))
        pl_module.log(
            "valid_loss",
            sum(self.valid_loss) / len(self.valid_loss)
        )
