from collections import Counter
from torchvision import datasets
import os
import json

word_freq = Counter()

train_dataset = datasets.CocoCaptions(
    root='coco2017/train2017',
    annFile='coco2017/annotations/captions_train2017.json',
)

val_dataset = datasets.CocoCaptions(
    root='coco2017/val2017',
    annFile='coco2017/annotations/captions_val2017.json',
)

dataset = [train_dataset, val_dataset]

for data in dataset:
    for img, texts in data:
        for text in texts:
            word_freq.update(text.lower().split())

print('Total words:', len(word_freq.keys()))

words = [w for w in word_freq.keys() if word_freq[w] > 5]

word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

print('Total tokens:', len(word_map))

with open(os.path.join('wordmap.json'), 'w') as j:
    json.dump(word_map, j)
