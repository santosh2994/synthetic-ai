import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
#%matplotlib inline
# read in all the words
words = open('names.csv', 'r').read().splitlines()[1:]
print(len(words))
print(max(len(w) for w in words))
print(words[:8])
words_orig = words
words = [w.lower() for w in words_orig]
print(words[:8])
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['*'] = 0
#stoi[';'] = 39
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)
# shuffle up the words
import random
random.seed(42)
random.shuffle(words)
# build the dataset
block_size = 8 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for ch in w + '*':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%
for x,y in zip(Xtr[:20], Ytr[:20]):
  print(''.join(itos[ix.item()] for ix in x), '-->', itos[y.item()])

torch.manual_seed(42); # seed rng for reproducibility

import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn((fan_in, fan_out)) / fan_in**0.5)  # note: kaiming init
        if bias:
            self.bias = nn.Parameter(torch.zeros(fan_out))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out

class BatchNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Parameters (trained with backprop)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # Buffers (trained with a running 'momentum update')
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

    def forward(self, x):
        # Calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)  # Batch mean
            xvar = x.var(dim, keepdim=True)  # Batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # Normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # Update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, IX):
        return self.embedding(IX)

class FlattenConsecutive(nn.Module):
    def __init__(self, n):
        super(FlattenConsecutive, self).__init__()
        self.n = n

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x

class Sequential(nn.Module):
    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Definition of the Tanh class
class Tanh(nn.Module):
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

import torch
import torch.nn as nn

# ... (the rest of your code)

class ReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)

# Now, you can create the model with ReLU activation
#n_embd = 30
#n_hidden = 512
#model = Sequential([
#    Embedding(vocab_size, n_embd),
#    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), ReLU(),
#    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), ReLU(),
#    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), ReLU(),
#    Linear(n_hidden, vocab_size),
#])


# Now, you can create the model as before
#n_embd = 30
#n_hidden = 512
#model = Sequential([
#    Embedding(vocab_size, n_embd),
#    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#    Linear(n_hidden, vocab_size),
#])


#%%time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define your custom layers here

# Define the model using the custom layers
class CustomModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden):
        super(CustomModel, self).__init__()
        # Create your model architecture using the custom layers
        self.model = Sequential([
            Embedding(vocab_size, n_embd),
            FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, vocab_size),
        ])

    def forward(self, x):
        return self.model(x)

# Initialize the model
n_embd = 30
n_hidden = 512
model = CustomModel(vocab_size, n_embd, n_hidden)

# Parameter initialization
with torch.no_grad():
    model.model.layers[-1].weight *= 0.1  # Last layer make less confident

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
model.to(device)

# Convert data to GPU
Xtr = Xtr.to(device)
Ytr = Ytr.to(device)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0001)
# Define the L2 regularization strength
#weight_decay = 1e-3

# Define the optimizer with weight decay (L2 regularization)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=weight_decay)

# Same optimization as last time
max_steps = 200000
batch_size = 128
lossi = []

for i in range(max_steps):
    # Minibatch construction
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), device=device)
    Xb, Yb = Xtr[ix], Ytr[ix]  # Batch X,Y

    # Forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update: simple SGD
    lr = 0.0001 if i < 150000 else 0.001  # Step learning rate decay
    optimizer.step()

    # Track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

'''for i in range(max_steps):
    # Minibatch construction
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), device=device)
    Xb, Yb = Xtr[ix], Ytr[ix]  # Batch X,Y

    # Forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # L2 regularization (weight decay)
    l2_regularization = 0.0
    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2) ** 2

    loss += weight_decay * l2_regularization

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update: simple SGD
    lr = 0.001 if i < 150000 else 0.01  # Step learning rate decay
    optimizer.step()

    # Track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())'''

model.eval()

# evaluate the loss
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
  x,y = {
    'train': (Xtr.to(device), Ytr.to(device)),
    'val': (Xdev.to(device), Ydev.to(device)),
    'test': (Xte.to(device), Yte.to(device)),
  }[split]
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')

import dill

# Save the model using dill.dump
with open('model.pkl', 'wb') as f:
    dill.dump(model, f)
