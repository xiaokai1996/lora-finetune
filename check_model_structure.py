
from transformers import AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen1.5-1.8B"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu")
    print(model)
except Exception as e:
    print(e)
