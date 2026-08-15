import torch
import torch.nn as nn
from torch.nn import functional as f

 
#parameter and numbers specified 
batch_size = 32
block_size =  8
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
device ='cuda' if torch.cuda.is_available() else 'cpu'
n_emb_d =32

with open("tinyshakespeare.txt", "r") as file:
    text = file.read()
    '''print(text)'''
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(" ".join(chars))
#print(vocab_size)

stoi= {s:i for i,s in enumerate(chars)}
itos ={i:s for i,s in enumerate(chars)}

encode = lambda s:[stoi[c] for c in s]
decode = lambda l: " ".join([itos[i] for i in l])


data = torch.tensor (encode(text), dtype =torch.long)
#print(data.shape ,data.type)
#print(data[0:1000])

n =int(0.9*len(data)) #training data set to 90 percent of total
training_data =data[:n]
validation_data =data[n:]

#print("training data:" , training_data)
#print("validation data:" , validation_data)


training_data[:block_size+1]

x = training_data[:block_size]
y = training_data[1:block_size+1]
for t in range(block_size):
   idx = x[:t+1]
   targets = y[t]
   # print(f"when input is {context} the target is :{target}")

 
torch.manual_seed(1337)

def get_batch(split):
    data = training_data if split == 'train' else validation_data
    ix = torch.randint(len(data)- block_size,(batch_size,)) # we are taking numbers from starting of the token list to the number - block size so that our code doesnt give invalid indices later in code
    x = torch.stack([data[i:i+block_size] for i in ix]) #  for training data we take from i to i+block size , which makes sure that we have 8 tokens 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #for target data 
    return x,y
xb,yb = get_batch('train')
'''print('inputs:')
print(xb.shape)
print(xb)
print('targets')
print(yb.shape)
print(yb)'''

torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X ,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 
class Head (nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key =nn.Linear(n_emb_d, head_size, bias=False)
        self.query =nn.Linear(n_emb_d,head_size,bias=False)
        self.value = nn.Linear(n_emb_d,head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C =x.shape
        k = self.key(x)
        q = self.query(x)

        wei =q@k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf'))
        wei = f.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads =nn.ModuleList([Head(head_size)for _ in range(num_heads)])
        self.proj = nn.Linear( n_emb_d, n_emb_d)
        self.droupout = nn.Dropout(dropout)
    def forward(self,x):
         out = torch.cat([h(x) for h in self.heads], dim=-1)
         out = self.droupout(self.proj(out))
         return out

class FeedForward(nn.Module):
 def __init__(self,n_emb_d):
     super().__init__()
     self.net = nn.Sequential(
         nn.Linear(n_emb_d, 4 * n_emb_d),
         nn.ReLU(),
          nn.Linear(4 * n_emb_d,n_emb_d),
          nn.Dropout(dropout), 
     )
 def forward(self,x):
     return self.net(x)

class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(0, keepdim=True) # batch mean
        xvar = x.var(0, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

#torch.manual_seed(1337)
#module = BatchNorm1d(100)
#x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
#x = module(x)
#x.shape

class Block(nn.Module):
    def __init__(self,n_emb_d , n_head):

        super().__init__()
        head_size = n_emb_d//n_head
        self.sa =MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_emb_d)
        self.ln1 = nn.LayerNorm(n_emb_d)
        self.ln2 = nn.LayerNorm(n_emb_d)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))# we added x to preserve the useful intial information to increase optimization throughout the blocks or layers of gpt
        x = x + self.ffwd(self.ln2(x))
        return x
     
    
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb_d)
        self.position_embedding_table = nn.Embedding(block_size, n_emb_d)
        self.blocks = nn.Sequential(*[Block(n_emb_d, n_head=n_head) for _ in range (n_layer)])
        self.ln_f = nn.LayerNorm(n_emb_d)
        self.lm_head = nn.Linear(n_emb_d, vocab_size) #here we have language model head to convert the embemmdings to the vocab size so we get token to predict



    def forward(self, idx, targets=None):

        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (b,t, vocab size)


        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = f.cross_entropy(logits, targets)

        return logits, loss

    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)
        return idx

    
model = BigramLanguageModel()
m = model.to(device)
logits,loss = model(xb,yb)  
out =  model(xb,yb)
#print(out.shape)  
context = torch.zeros((1,1),dtype=torch.long)
#print(decode(model.generate(context,max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for step in range(max_iters):
    xb , yb = get_batch('train')
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context = torch.zeros((1,1),dtype=torch.long, device=device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))