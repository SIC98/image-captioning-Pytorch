import torch


def encode_texts_2d(texts_2d, tokenizer):
    # Flatten the list of lists and encode texts
    texts_2d = [list(t) for t in zip(*texts_2d)]
    encoded_texts = [encode_texts(texts, tokenizer)[0] for texts in texts_2d]

    # Finding the maximum length
    max_len = max(len(text) for sublist in encoded_texts for text in sublist)

    # Padding
    for sublist in encoded_texts:
        for text in sublist:
            if len(text) < max_len:
                text += [50256] * (max_len - len(text))

    return encoded_texts


def encode_texts(texts, tokenizer):
    # Encoding texts
    encoded_texts = [encode_text(text, tokenizer) for text in texts]

    # Finding the maximum length
    texts_length = [len(text) for text in encoded_texts]
    max_len = max(texts_length)

    # Padding
    for text in encoded_texts:
        if len(text) < max_len:
            text += [50256] * (max_len - len(text))

    texts_length = [[text_length] for text_length in texts_length]

    return encoded_texts, texts_length


def encode_text(text, tokenizer):

    # Adding "<start>" and "<end>" tokens and encoding the words
    words_encoded = tokenizer.encode(text)

    words_encoded = [50256] + \
        words_encoded + [50256]

    return words_encoded


def collate_fn(samples):
    images, captions = zip(*samples)

    images = torch.stack(images, 0)

    max_length = max(len(c) for c in captions)
    padded_captions = []
    for c in captions:
        if len(c) < max_length:
            c += [''] * (max_length - len(c))
        padded_captions.append(c)

    return images, padded_captions
