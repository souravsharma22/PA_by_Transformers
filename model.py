import tensorflow as tf
import torch
import torch.nn as nn

#GELU Activation function
class GELU(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.0447*torch.pow(x,3))))
    
#Feed Forward Neural network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']), #incresing dimesion
                                    GELU(),
                                    nn.Linear(4*cfg['emb_dim'],cfg['emb_dim']))  #coming back to original dimension
    def forward(self,x):
        return self.layers(x)
    
# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, emd_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emd_dim))
        self.shift = nn.Parameter(torch.zeros(emd_dim))
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim =True)
        var = x.var(dim =-1, keepdim = True, unbiased= False)
        norm_x = (x-mean)/(torch.sqrt(var+self.eps))
        return self.scale*norm_x + self.shift
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads==0),\
            "d_out must be divisible by nums head"
        
        self.d_out = d_out
        self.num_head = num_heads
        self.head_dim = d_out//num_heads
        # self.d_in = d_in
        self.w_query = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in,d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b,num_token,d_in = x.shape

        keys = self.w_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.w_query(x)
        values = self.w_value(x)

        keys = keys.view(b,num_token,self.num_head,self.head_dim)
        queries = queries.view(b,num_token,self.num_head,self.head_dim)
        values = values.view(b,num_token,self.num_head,self.head_dim)

        #grouping by num_heads
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        # calculating attention score
        attn_score = queries @ keys.transpose(2,3)

        # calculating attention weigths,masking, scaling and dropout
        mask_bool = self.mask.bool()[:num_token,:num_token]
        attn_score= attn_score.masked_fill_(mask_bool, - torch.inf)
        attn_weight = torch.softmax(attn_score/keys.shape[-1]**0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)
        
        #calculating the context vector
        context_vector = attn_weight @ values #ntokn x ntoken * ntoken x head_dim
        # trasposing to get all the context vextor togeth
        context_vector = context_vector.transpose(1,2)

        # combining heads 
        context_vector = context_vector.contiguous().view(b,num_token,self.d_out)
        context_vector = self.out_proj(context_vector) # optional projection
        return context_vector
    
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MultiHeadAttention(d_in=cfg['emb_dim'], d_out= cfg['emb_dim'],
                                       context_length=cfg['context_length'],
                                       num_heads=cfg['n_heads'], dropout= cfg['drop_rate'],
                                       qkv_bias= cfg['qkv_bias'])
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(emd_dim=cfg['emb_dim'])
        self.norm2 = LayerNorm(emd_dim=cfg['emb_dim'])

        self.drop_shortcut  = nn.Dropout(cfg['drop_rate'])
    
    def forward(self,x):
        shortcut = x
        x= self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)

        x = x+ shortcut

        shortcut =x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm  = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'],cfg['vocab_size'], bias=False)
    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        token_embded = self.tok_emb(in_idx)
        pos_embded = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        x = token_embded+pos_embded
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

