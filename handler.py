"""
RunPod handler that

1. Receives a JSON payload
   {
     "adapter_name": "my_first_lora",          # required
     "bucket":       "my-lora-bucket",         # required
     "prefix":       "experiments",            # defaults to 'adapters'
     "prompt":       "Explain quantum gravity in simple words.",
     "max_new_tokens": 128                     # optional
   }

2. Downloads the LoRA adapter files from s3://bucket/prefix/adapter_name/*
3. Attaches the adapter to the base model (Mistral‑7B‑Instruct‑v0.2)
4. Generates a completion and returns
   { "status": "ok", "text": "<assistant reply>" }
"""

import os, tempfile, boto3, json, gc, torch, runpod
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import login as hf_login

load_dotenv()
s3 = boto3.client("s3")

# ------------------------------------------------------------------ helpers
def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def download_adapter(bucket: str, key_prefix: str) -> str:
    tmp = tempfile.mkdtemp()
    for fname in ("adapter_model.bin", "adapter_config.json",
                  "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json"):
        key = f"{key_prefix}/{fname}"
        local = os.path.join(tmp, fname)
        try:
            s3.download_file(bucket, key, local)
        except s3.exceptions.NoSuchKey:
            # ignore optional files
            continue
    return tmp



# ------------------------------------------------------------------ handler
def handler(event):
    free_gpu()                        # clean slate

    inp = event.get("input", {})
    bucket       = inp["bucket"]
    adapter_name = inp["adapter_name"]
    prefix       = inp.get("prefix", "adapters").rstrip("/")
    prompt       = inp.get("prompt", "Hello!")
    max_tokens   = int(inp.get("max_new_tokens", 128))

    adapter_s3_prefix = f"{prefix}/{adapter_name}"
    adapter_dir       = download_adapter(bucket, adapter_s3_prefix)

    # login to HF (so the base model comes from the private repo cache if needed)
    hf_login(os.getenv("HUGGINGFACE_TOKEN"))

    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # move off‑GPU & flush
    model.to("cpu")
    free_gpu()

    return {"status": "ok", "text": text}

# ----- entry‑point for RunPod --------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
