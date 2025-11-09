import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import ctranslate2
import os

ADAPTER_PATH = "./models" 
OUTPUT_CT2_PATH = "./models_ct2"
BASE_MODEL_NAME = "openai/whisper-small"

print(f"Loading base model '{BASE_MODEL_NAME}'...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
)
base_model.eval()


print(f"Loading and merging adapter from '{ADAPTER_PATH}'...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
merged_model = peft_model.merge_and_unload()
print("Model merged successfully.")

print(f"Converting merged model to CTranslate2 format (quantization=int8)...")
converter = ctranslate2.converters.TransformersConverter(merged_model)
converter.convert(
    output_dir=OUTPUT_CT2_PATH,
    quantization="int8", 
    force=True, 
)

print(f"Copying processor/tokenizer files to '{OUTPUT_CT2_PATH}'...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)
processor.save_pretrained(OUTPUT_CT2_PATH)

print("\n🎉 Conversion complete!")
print(f"Your CTranslate2 optimized model is ready in: {OUTPUT_CT2_PATH}")