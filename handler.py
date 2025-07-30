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
    """
    Pull every object under the S3 prefix into a tmp dir and
    return that directory path.
    """
    tmp_dir = tempfile.mkdtemp()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = os.path.join(tmp_dir, os.path.relpath(key, key_prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
    return tmp_dir

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
