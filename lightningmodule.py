import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import json

from utils import encode_texts


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module
    ):
        super().__init__()
        self.model = model

        self.loss = nn.CrossEntropyLoss()

        with open('wordmap.json', 'r') as j:
            self.word_map = json.load(j)

    def training_step(self, batch, batch_idx):  # optimizer_idx
        img, text, caplens = batch

        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.model(
            img, text, caplens
        )

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # Update code by using: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/86#issuecomment-539709462
        scores = pack_padded_sequence(
            scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True).data

        loss = self.loss(scores, targets)

        alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        return {
            'loss': loss,
        }

    def validation_step(self, batch, batch_idx):
        self.model.eval()  # eval mode (no dropout or batchnorm)

        img, text, caplens = batch

        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.model(
            img, text, caplens
        )

        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(
            scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True).data

        loss = self.loss(scores, targets)

        alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        return {
            'loss': loss,
        }

    def on_before_batch_transfer(self, batch, dataloader_idx):
        img, text = batch

        tokenized_text, caplens = encode_texts(text[0], self.word_map)

        return img, torch.tensor(tokenized_text, device=self.device), torch.tensor(caplens, device=self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-4,
        )
        return optimizer
        # encoder_optimizer = torch.optim.AdamW(
        #     params=self.model.encoder.parameters(),
        #     lr=1e-4,
        # )
        # decoder_optimizer = torch.optim.AdamW(
        #     params=self.model.decoder.parameters(),
        #     lr=4e-4,
        # )
        # return [encoder_optimizer, decoder_optimizer]
