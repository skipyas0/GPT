import torch
from model import TransformerDecoder
from utils import *

if __name__ == "__main__":
    # torch.manual_seed(1337)
    contents = load_text("input.txt")
    chars, vocab_size = get_chars(contents)
    encode, decode = get_encode_decode(chars)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerDecoder(vocab_size)

    model.load_state_dict(torch.load("weights_shakespeare", weights_only=True))
    model.to(device)
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen = model.generate(idx, max_new_tokens=10000)

    with open("ai_shakespeare2.txt", "w") as f:
        f.write(decode(gen[0].tolist()))
