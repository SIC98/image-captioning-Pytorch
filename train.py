import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2Tokenizer, GPT2Model

from datamodules import COCODataModule
from lightningmodule import LightningModule
from models.models import EncoderDecoder

from loggers import OutputLogger
import wandb


pl.seed_everything(42)

model = EncoderDecoder(
    attention_dim=512,
    embed_dim=512,
    decoder_dim=512,
    vocab_size=12982,
    encoder_dim=2048,
    dropout=0.5
)

gpt2 = GPT2Model.from_pretrained('gpt2')
model.load_pretrained_embeddings(gpt2.wte)
del gpt2

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
            dirpath=wandb.run.dir,
            save_top_k=-1,
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
