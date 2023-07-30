def encode_texts(texts, word_map):
    # Encoding texts
    encoded_texts = [encode_text(text, word_map) for text in texts]

    # Finding the maximum length
    max_len = max(len(text) for text in encoded_texts)

    # Padding
    for text in encoded_texts:
        if len(text) < max_len:
            text += [word_map['<pad>']] * (max_len - len(text))

    return encoded_texts


def encode_text(text, word_map):
    words = text.split(" ")

    # Adding "<start>" and "<end>" tokens and encoding the words
    words_encoded = [word_map.get(
        word, word_map['<unk>']) for word in words]

    words_encoded = [word_map['<start>']] + \
        words_encoded + [word_map['<end>']]

    return words_encoded
