import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datamodules import COCODataModule
from lightningmodule import LightningModule
from models.models import EncoderDecoder

from loggers import OutputLogger


pl.seed_everything(42)

model = EncoderDecoder(
    attention_dim=512,
    embed_dim=512,
    decoder_dim=512,
    vocab_size=12982,
    encoder_dim=2048,
    dropout=0.5
)

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
        OutputLogger(),
    ],
    max_epochs=10,
)

trainer.fit(
    model=lightningmodule,
    datamodule=datamodule,
)
