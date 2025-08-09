import numpy as np
import torch
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left = {left.shape}, right = {right.shape}")
    with torch.no_grad():
        left.copy_(torch.tensor(right))
    return left

def load_weights_to_gpt(gpt,params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight,params['wpe'] )
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    # loading weights to transformer blocks and it's each layer
    for b in range(len(params['blocks'])):
        #upadting key,query and value weights and bias
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.w_query.weight = assign(
            gpt.trf_blocks[b].attn.w_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.w_key.weight = assign(
            gpt.trf_blocks[b].attn.w_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.w_value.weight = assign(
            gpt.trf_blocks[b].attn.w_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.w_query.bias = assign(
            gpt.trf_blocks[b].attn.w_query.bias, q_b)
        gpt.trf_blocks[b].attn.w_key.bias = assign(
            gpt.trf_blocks[b].attn.w_key.bias, k_b)
        gpt.trf_blocks[b].attn.w_value.bias = assign(
            gpt.trf_blocks[b].attn.w_value.bias, v_b)
        
        #updating out projection weights and bias
        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        # updating weight of feed forward neural network 
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        # updating weights for layer normalization scale and shift
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])
    # End of transformer blocks
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params['g'])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params['b'])
    gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])