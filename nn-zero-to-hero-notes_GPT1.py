#source: https://github.com/MK2112/nn-zero-to-hero-notes/blob/main/gpt.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import nn, device, manual_seed, no_grad

from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)

from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog, convert_to_digital
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, IdealDevice, FloatingPointDevice
#from torch import cuda
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import ReRamSBPreset, ReRamESPreset
#import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=32'
torch.cuda.empty_cache()

USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")
device=DEVICE

# Hyperparameters
batch_size = 64      # How many independent sequences to process at once?
batch_size = 19      # How many independent sequences to process at once?
block_size = 256     # What is the maximum context length for predictions?
#block_size = 16     # What is the maximum context length for predictions?
max_iters = 3     # How many training iterations to run?
eval_interval = 1  # How often to evaluate the model on the validation set?
learning_rate = 3e-4 # Learning rate for Adam optimizer (found through trial and error)
#device = 'cuda' if torch.cuda.is_available() else 'cpu' # Don't run on CPU if possible (it's slow. really.)
eval_iters = 200     # How many batches to use per loss evaluation?
n_embd = 384         # Number of hidden units in the Transformer (384/6 = 64 dimensions per head)
#n_embd = 64         # Number of hidden units in the Transformer (384/6 = 64 dimensions per head)
n_head = 6           # Number of attention heads in a single Transformer layer
#n_head = 2           # Number of attention heads in a single Transformer layer
n_layer = 6          # Number of Transformer layers
#n_layer = 1          # Number of Transformer layers
dropout = 0.2        # Dropout probability

torch.manual_seed(1337)

# Load Tiny Shakespeare dataset 
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# (also refer to Andrej Karpathy's blog: http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Find all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings from characters to indices and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]          # encoder: Take a string, return a list of indices/integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: Take a list of indices/integers, return a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% of all characters are for training
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generates a tensor of shape (batch_size,) with random sequence start indices between 0 and len(data) - block_size
    x = torch.stack([data[i:i+block_size] for i in ix])       # Stack all (ix holds batch_size many) sequences of this batch row-wise on top of each other to form a tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # Same as x but shifted by one token
    x, y = x.to(device), y.to(device)
    return x, y # x is batch_size x block_size, y is batch_size x block_size

@torch.no_grad() # Disable gradient calculation for this function
def estimate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Register a buffer so that it is not a parameter of the model

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape   # Batch size, block size, vocab size (each token is a vector of size 32)
        k = self.key(x)   # (B,T,C) -> (B,T, head_size)
        q = self.query(x) # (B,T,C) -> (B,T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5                       # (B, T, head_size) @ (B, head_size, T) = (B, T, T) (T is the block_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking all values in wei where tril == 0 with -inf
        wei = F.softmax(wei, dim=-1)                                 # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        v = self.value(x) # (B, T, C) -> (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Create num_heads many heads
        self.proj = nn.Linear(n_embd, n_embd)                                   # Projecting back to n_embd dimensions (the original size of the input, because we use residual connections)
        self.dropout = nn.Dropout(dropout)                                      # Dropout layer for regularization

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate the outputs of all heads
        out = self.dropout(self.proj(out))                  # Project back to n_embd dimensions (because we use residual connections) and apply dropout
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Linear layer with 4*n_embd outputs (AIAYN suggests 4*n_embd for residual connections as channel size)
            nn.ReLU(),                     # ReLU introduces non-linearity
            nn.Linear(4 * n_embd, n_embd), # Linear layer with n_embd outputs
            nn.Dropout(dropout),           # Dropout layer for regularization
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head                    # Adapting the head size to the number of heads
        self.sa = MultiHeadAttention(n_head, head_size) # Self-attention multi-head layer (the communication)
        self.ffwd = FeedFoward(n_embd)                  # Feed-forward so that the output has the same dimension as the input (the computation)
        self.ln1 = nn.LayerNorm(n_embd)                 # Layer normalization (normalizes the output of the self-attention layer)
        self.ln2 = nn.LayerNorm(n_embd)                 # Layer normalization (normalizes the output of the feed-forward layer)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))                    # Residual connection, forking off to the self-attention layer, LayerNorm is applied before the self-attention layer
        x = x + self.ffwd(self.ln2(x))                  # Residual connection, forking off to the feed-forward layer, LayerNorm is again applied before the feed-forward layer
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embd = nn.Embedding(vocab_size, n_embd)                                   # Embedding the vocabulary, each individual token is represented by a vector of size vocab_size x n_embd
        self.position_embd = nn.Embedding(block_size, n_embd)                                # Embedding the position, each position is represented by a vector of size block_size x n_embd
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)                                         # Linear layer to map the embedding to the vocabulary size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embd(idx)                                                      # Embedding the input, shape is (batch_size, block_size, n_embd) (B, T, n_embd)
        pos_embd = self.position_embd(torch.arange(T, device=device))                        # Embedding the position by providing an integer sequence up to block_size, shape is (block_size, n_embd) (T, n_embd)
        x = tok_embd + pos_embd                                                              # Adding the token embedding and the position embedding, shape is (batch_size, block_size, n_embd) (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                             # Calculating the logits, shape is (batch_size, block_size, vocab_size) (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)                # Transpose logits to (B, C, T) (B=batch_size, T=block_size, C=vocab_size)
            targets = targets.view(B*T)                 # Transpose targets to (B, T)
            loss = F.cross_entropy(logits, targets)     # Calculating cross entropy loss across all tokens in the batch

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                    # Condition on the last block_size tokens (B, T)
            logits, _ = self(idx_cond)                         # Forward pass (this is the forward function) with the current sequence of characters idx, results in (B, T, C)
            logits = logits[:, -1, :]                          # Focus on the last token from the logits (B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # Calculate the set of probabilities for the next token based on this last token, results in (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # Sample the next token (B, 1), the token with the highest probability is sampled most likely
            idx = torch.cat((idx, idx_next), dim=1)            # Add the new token to the sequence (B, T+1) for the next iteration
        return idx

def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]                    # Condition on the last block_size tokens (B, T)
        logits, _ = self(idx_cond)                         # Forward pass (this is the forward function) with the current sequence of characters idx, results in (B, T, C)
        logits = logits[:, -1, :]                          # Focus on the last token from the logits (B, T, C) -> (B, C)
        probs = F.softmax(logits, dim=-1)                  # Calculate the set of probabilities for the next token based on this last token, results in (B, C)
        idx_next = torch.multinomial(probs, num_samples=1) # Sample the next token (B, 1), the token with the highest probability is sampled most likely
        idx = torch.cat((idx, idx_next), dim=1)            # Add the new token to the sequence (B, T+1) for the next iteration
    return idx



# Model
model = BigramLanguageModel()
#print(model)
#print("\n \n \n \n")
rpu_config = SingleRPUConfig(device=ConstantStepDevice())
#rpu_config = SingleRPUConfig(device=IdealDevice())
#rpu_config = SingleRPUConfig(device=FloatingPointDevice())
model = convert_to_analog(model, rpu_config)
#model.remap_analog_weights()
#print(model)
#model = BigramLanguageModel()
#model = model.to(device)
# Prepare the model.
if USE_CUDA:
    model = model.cuda()
#print(model)
#print(model)
#print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters') # print the number of parameters in the model

# Create a PyTorch optimizer
#opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
opt = AnalogAdam(model.parameters(), lr=learning_rate)
opt.regroup_param_groups(model)

# Training loop
for iter in range(max_iters):
#    print("working1")
    if iter % eval_interval == 0 or iter == max_iters - 1:
#        print("working2a")
        #Additional Info when using cuda        
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')            
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#    print("working2b")
    xb, yb = get_batch('train')     # Get batch
#    print("working3")
    logits, loss = model(xb, yb)    # Forward pass
#    print("working4")
    opt.zero_grad(set_to_none=True) # Reset gradients
#    print("working5")
    loss.backward()                 # Backward pass
#    print("working6")
    opt.step()                      # Update parameters
#    print("working7")
    # Save the model architecture
    if iter % 1000 == 0:
        torch.save(model, f"model_{iter}.pt")
    del loss, logits
    #torch.cuda.empty_cache() 

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)     # Start with single token as context
model.generate = generate.__get__(model)
#print(model)
#model = convert_to_digital(model)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters') # print the number of
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
