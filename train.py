import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datamodules import COCODataModule
from lightningmodule import LightningModule
from models.models import EncoderDecoder


pl.seed_everything(42)

model = EncoderDecoder()
lightningmodule = LightningModule(model=model)

datamodule = COCODataModule()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=-1,
    logger=[
        WandbLogger(project='Image Captioning'),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor='valid_loss',
            save_last=True,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval='step'),
    ],
    max_epochs=10,
)

trainer.fit(
    model=lightningmodule,
    datamodule=datamodule,
)
