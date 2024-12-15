"""
the transformer neural-architecture initially developed for language processing seems to hit at a 
deep truth about complex/context-dependant/flexible data structures such as: human language, morphogenesis
"""

with open("training_data.txt", "r", encoding="utf-8") as f:
    text = f.read()
print("length of text: ", len(text))


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)


# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string
print(encode("hii there"))
print(decode(encode("hii there")))


# let's now encode the entire text dataset and store it into a torch.Tensor
import torch  # we use PyTorch: https://pytorch.org

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(
    data[:1000]
)  # the 1000 characters we looked at earier will to the GPT look like this


# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


block_size = 8
train_data[: block_size + 1]


x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    # ????
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)
print("----")

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")
