import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
def token_to_text(ids, tokenizer):
    flat = ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
def generate(model,idx,max_new_token,context_size,temp=0.0,top_k=None,eos_id= None):
    for _ in range(max_new_token):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        # applying filter for top k
        if top_k is not None:
            top_logits,_  = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(logits<min_val,torch.tensor(float("-inf")).to(device),logits)
        # Apply temprature scaling
        if temp>0.0:
            logits = logits/temp
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next==eos_id:
            break
        idx = torch.cat((idx,idx_next),dim=-1)

    return idx
