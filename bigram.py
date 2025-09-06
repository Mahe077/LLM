import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 #how many independent sequences will we process in parallel? 
block_size = 256 #what is the maximum context length for predictions? (i.e. how many characters do we feed into the model to predict the next character?)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------------------------

torch.manual_seed(1337) #for reproducibility 👉 Without torch.manual_seed(...), each run would give different random values.

# curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#read in all the text from tinyshakespeare.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # 👉 \n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab_size = len(chars) #the number of unique characters in the text
# print("".join(chars))

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } #string to integer mapping by enumerating over the chars {'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, ...}
itos = { i:ch for i,ch in enumerate(chars) } #integer to string mapping by enumerating over the chars {0: '\n', 1: ' ', 2: '!', 3: '$', 4: '&', ...}

encode = lambda s: [stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

# print(encode("hello world")) #example [46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]
# print(decode(encode("hello world"))) #example "hello world"

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) #convert the entire text into a tensor of integers 👉 tensor([18, 47, 56,  ..., 43, 56, 43])
n = int(0.9*len(data)) #first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data #choose the correct dataset
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 👉 ix is a tensor of random starting positions for your training sequences.
    x = torch.stack([data[i:i+block_size] for i in ix]) #for each starting index, grab a sequence of length block_size
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #the target is the same sequence but shifted to the right by one character
    x, y = x.to(device), y.to(device) #move to device (GPU or CPU)
    #If i = 76049,
    #   x will be characters [76049 : 76057]
    #   y will be characters [76050 : 76058]
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) #tensor to hold the losses for each iteration
        for k in range(eval_iters):
            X, Y = get_batch(split) #get a batch of data
            logits, loss = model(X, Y) #forward pass
            losses[k] = loss.item() #record the loss
        out[split] = losses.mean() #average loss over all iterations
    model.train() #set the model back to training mode
    return out

# Head
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,C)
        q = self.query(x)   # (B,T,C
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # softmax the attention scores
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #expand the embedding dimension by 4x
            nn.ReLU(), #non-linearity
            nn.Linear(4 * n_embd, n_embd), #project back to the original embedding dimension
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, head_size: the dimension of each head
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # i.e. 4 heads of 8-dimensional self-attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # (B,T,C)
        x = x + self.ffwd(self.ln2(x)) # (B,T,C)
        return x

# Define a simple Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #embedding table to convert token indices to vectors of size n_embd
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) #embedding table to convert position indices to vectors of size n_embd
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #linear layer to project the embeddings to the vocabulary size

    def forward(self, idx, targets=None):
        B, T = idx.shape #batch size and sequence length

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos = torch.arange(T, device=device) # (T,) tensor of integers from 0 to T-1
        pos_emb = self.positional_embedding_table(pos) # (T,C)
        x = token_emb + pos_emb # (B,T,C) -> This is broadcasting (T,C) to (B,T,C) # x holds the token embeddings and positional embeddings
        x = self.blocks(x) # (B,T,C) # apply the transformer blocks
        x = self.ln_f(x) # (B,T,C) # apply final layer norm
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape for cross-entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, block_size)
            # get the predictions
            logits, _ = self(idx_cond) # (B, block_size, vocab_size)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Instantiate the model and move to device
model = BigramLanguageModel()
model = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Iteration {iter}: Train loss {losses['train']:}, Val loss {losses['val']}")


    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)

    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # start with a single zero token
print(decode(model.generate(context, max_new_tokens=500)[0].tolist())) #generate

# Save model
torch.save(model.state_dict(), "shakespeare_model.pt")

# # Recreate model architecture
# model = BigramLanguageModel().to(device)
# model.load_state_dict(torch.load("shakespeare_model.pt", map_location=device))
# model.eval()