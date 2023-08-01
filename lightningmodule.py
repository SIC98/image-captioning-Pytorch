import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import json
import wandb
import random

from utils import encode_texts, encode_texts_2d


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module
    ):
        super().__init__()
        self.model = model
        self.automatic_optimization = False
        self.validation_step_outputs = []

        self.loss = nn.CrossEntropyLoss()

        with open('wordmap.json', 'r') as j:
            self.word_map = json.load(j)

    def training_step(self, batch, batch_idx):  # optimizer_idx
        encoder_opt, decoder_opt = self.optimizers()
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        img, cap, allcaps, caplens = batch

        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.model(
            img, cap, caplens
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

        self.manual_backward(loss)
        encoder_opt.step()
        decoder_opt.step()

        return {
            'loss': loss,
        }

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        img, cap, allcaps, caplens = batch

        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.model(
            img, cap, caplens
        )

        targets = caps_sorted[:, 1:]

        scores_copy = scores.clone()
        scores = pack_padded_sequence(
            scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True).data

        loss = self.loss(scores, targets)

        alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        references = list()
        hypotheses = list()

        # References
        allcaps = allcaps[sort_ind]
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>'], self.word_map['<end>']}],
                    img_caps))  # remove <start> and pads

            # Remove empty reference
            img_captions = [
                not_empty_str for not_empty_str in img_captions if not_empty_str != []
            ]
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

        return {
            'loss': loss,
            'references': references,
            'hypotheses': hypotheses
        }

    def on_before_batch_transfer(self, batch, dataloader_idx):
        img, allcaps = batch

        # Delete empty caption
        cap = [
            random.choice(
                [not_empty_str for not_empty_str in c if not_empty_str != '']
            ) for c in allcaps
        ]

        tokenized_cap, caplens = encode_texts(cap, self.word_map)
        tokenized_allcaps = encode_texts_2d(allcaps, self.word_map)

        return img, \
            torch.tensor(tokenized_cap, device=self.device), \
            torch.tensor(tokenized_allcaps, device=self.device), \
            torch.tensor(caplens, device=self.device)

    def configure_optimizers(self):
        encoder_optimizer = torch.optim.AdamW(
            params=self.model.encoder.parameters(),
            lr=1e-4,
        )
        decoder_optimizer = torch.optim.AdamW(
            params=self.model.decoder.parameters(),
            lr=4e-4,
        )
        return [encoder_optimizer, decoder_optimizer]
