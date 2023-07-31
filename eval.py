import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

from models.models import EncoderDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncoderDecoder(
    attention_dim=512,
    embed_dim=512,
    decoder_dim=512,
    vocab_size=12982,
    encoder_dim=2048,
    dropout=0.5
)

checkpoint = torch.load(
    'wandb/run-20230731_114904-7twg4vj4/files/epoch=5-step=44352.ckpt'
)

new_state_dict = OrderedDict()
for n, v in checkpoint['state_dict'].items():
    name = n.replace("model.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)

model.eval()

decoder = model.decoder
encoder = model.encoder

with open('wordmap.json', 'r') as j:
    word_map = json.load(j)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.CocoCaptions(
        root='coco2017/val2017',
        annFile='coco2017/annotations/captions_val2017.json',
        transform=transform
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        drop_last=False,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    for i, (image, allcaps) in enumerate(
        tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))
    ):

        cap = [c[0] for c in allcaps]

        tokenized_cap, caplens = encode_texts(cap, word_map)
        tokenized_allcaps = encode_texts_2d(allcaps, word_map)

        cap = torch.tensor(tokenized_cap, device=device)
        allcaps = torch.tensor(tokenized_allcaps, device=device)
        caplens = torch.tensor(caplens, device=device)

        k = beam_size

        image = image.to(device)

        # Encode
        # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        # (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.reshape(
            1, -1, encoder_dim)  # view -> reshape
        num_pixels = encoder_out.size(1)

        # (k, num_pixels, encoder_dim)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)

            # (s, encoder_dim), (s, num_pixels)
            awe, _ = decoder.attention(encoder_out, h)

            # gating scalar, (s, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe

            h, c = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.reshape(  # view -> reshape
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        if complete_seqs_scores != []:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

            # Hypotheses
            hypotheses.append([w for w in seq if w not in {
                word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 5
    print(f"BLEU-4 score @ beam size of {beam_size} is {evaluate(beam_size)}")
