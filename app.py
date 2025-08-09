from flask import Flask, render_template, request

import torch
import tiktoken
from model import GPTModel
from loadweight_to_model import load_weights_to_gpt 
from weight_down import download_and_load_gpt2
from utility import text_to_token, token_to_text , generate 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = tiktoken.get_encoding('gpt2')

NEW_CONFIG = {
    'vocab_size': 50257,
 'context_length': 1024,
 'emb_dim': 1024,
 'n_heads': 16,
 'n_layers': 24,
 'drop_rate': 0.1,
 'qkv_bias': True
 }

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task"
        f"Write a instruction that appropriatly completes the request"
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    return instruction_text+input_text



app = Flask(__name__)


#loading model
@app.before_first_request
def load_model():
    global PA_model, optimizer
    print("Loading model...")
    PA_model = GPTModel(NEW_CONFIG)
    optimizer = torch.optim.AdamW(PA_model.parameters(), lr=0.00005, weight_decay=0.1)
    checkpoint = torch.load("checkpoints/gpt_checkpoint_epoch_3.pth", map_location=device)
    PA_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    PA_model.to(device)
    PA_model.eval()
    print("Model loaded.")


@app.route("/", methods=["GET", "POST"])
def form_page():
    instruction = None
    user_input = None
    final_input = None
    output = None
    if request.method == "POST":
        instruction = request.form.get("instruction")
        user_input = request.form.get("input")
        data = {
            "instruction": instruction,
            "input": user_input
        }
        # genrating output
        final_input = format_input(data)
        output_ids = generate(model=PA_model, idx= text_to_token(final_input,tokenizer), max_new_token=30,
                     context_size=NEW_CONFIG['context_length'],eos_id=50256)
        output = token_to_text(output_ids,tokenizer)
    if output is None:
        model_output = ""
    else:
        model_output = output.split("### Response:")[-1].strip()    

    return render_template("index.html",
                           instruction= instruction, user_input = user_input, model_output = model_output)

if __name__ == "__main__":
    app.run(debug=True)
