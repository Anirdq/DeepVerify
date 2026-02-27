"""
DeepVerify – One-time model downloader
Run this ONCE with internet access, then use uvicorn normally.

    python download_model.py
"""
import os
import sys

# Must be set BEFORE any huggingface import
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Downloading GPT-2 tokenizer and model to local cache...")
print("(Forcing legacy .bin format to bypass XetHub CDN)\n")

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("[1/2] Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("      ✓ Tokenizer cached.\n")

    print("[2/2] Model weights (pytorch_model.bin via standard HTTPS)...")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        use_safetensors=False,   # <-- forces .bin, avoids XetHub entirely
    )
    print("      ✓ Model cached.\n")

    print("=" * 55)
    print("SUCCESS! Start the backend with:")
    print("  uvicorn main:app --reload --port 8000")
    print("=" * 55)

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nYou can still run the backend (heuristics-only mode):")
    print("  uvicorn main:app --reload --port 8000")
    sys.exit(1)
