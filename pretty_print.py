

from model import ModelArgs, Transformer

args = ModelArgs(
    vocab_size=32000,
    device='cpu'
)

model = Transformer(args)
print(model)

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")
