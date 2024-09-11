import torch
from model import TransformerDecoder
from utils import *


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    # hyperparams
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 300
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 200
    n_embed = 384
    n_heads = 6
    num_blocks = 8
    ff_multiplier = 6
    dropout = 0.2
    # -------
    torch.manual_seed(1337)

    contents = load_text("input.txt")

    chars, vocab_size = get_chars(contents)

    encode, decode = get_encode_decode(chars)

    data = torch.tensor(encode(contents), dtype=torch.long)

    n = int(0.9 * len(data))

    train_data = data[:n]

    val_data = data[n:]

    model = TransformerDecoder(vocab_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 32
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "weights_messages")
