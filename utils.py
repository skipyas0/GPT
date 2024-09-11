def load_text(path):
    with open(path, "r") as f:
        contents = f.read()
    return contents


def get_chars(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return chars, vocab_size


def get_encode_decode(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    return encode, decode
