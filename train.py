import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2Model, GPT2Tokenizer
import wandb

from datamodules import COCODataModule
from lightningmodule import LightningModule
from models.models import EncoderDecoder
from loggers import OutputLogger


pl.seed_everything(42)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2")

model = EncoderDecoder(
    attention_dim=512,
    embed_dim=gpt2.wte.weight.shape[1],
    decoder_dim=512,
    vocab_size=tokenizer.vocab_size,
    encoder_dim=2048,
    dropout=0.5
)

model.decoder.load_pretrained_embeddings(weight=gpt2.wte.weight, freeze=True)

lightningmodule = LightningModule(
    model=model, train_encoder=True, train_decoder=True
)

datamodule = COCODataModule()

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    logger=[
        WandbLogger(project="Image Captioning"),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor="valid_loss",
            dirpath=wandb.run.dir,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="valid_loss", mode="min", patience=3),
        OutputLogger(),
    ],
    max_epochs=-1,
)

trainer.fit(
    model=lightningmodule,
    datamodule=datamodule,
)
