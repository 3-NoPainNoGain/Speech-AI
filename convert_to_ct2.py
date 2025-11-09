import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import ctranslate2
import os
import shutil  

MODEL_PATH = "./models" 

OUTPUT_CT2_PATH = "./models_ct2"
BASE_MODEL_NAME = "openai/whisper-small"
TEMP_MERGED_PATH = "./models_merged_temp"

def convert_model():

    print(f"'{BASE_MODEL_NAME}'의 Processor를 로드합니다...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)
    
    print(f"'{BASE_MODEL_NAME}'의 원본 모델(float32)을 로드합니다...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32, # CPU에서 안전하게 병합하기 위해 float32 사용
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    print(f"'{MODEL_PATH}'에서 LoRA 어댑터를 로드하고 원본 모델과 병합합니다...")
    peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    merged_model = peft_model.merge_and_unload()
    print("모델 병합 완료.")

    print(f"병합된 모델을 임시 폴더 '{TEMP_MERGED_PATH}'에 저장합니다...")
    merged_model.save_pretrained(TEMP_MERGED_PATH)
    
    print(f"임시 모델을 CTranslate2 포맷으로 변환합니다 (대상: '{OUTPUT_CT2_PATH}')...")
    converter = ctranslate2.converters.TransformersConverter(TEMP_MERGED_PATH)

    converter.convert(
        output_dir=OUTPUT_CT2_PATH,
        quantization="int8", 
        force=True, # 이미 폴더가 있어도 덮어쓰기
    )

    print(f"서버가 사용할 Processor 파일들을 '{OUTPUT_CT2_PATH}'에 저장합니다...")
    processor.save_pretrained(OUTPUT_CT2_PATH)

    try:
        shutil.rmtree(TEMP_MERGED_PATH)
        print(f"임시 폴더 '{TEMP_MERGED_PATH}'를 삭제했습니다.")
    except OSError as e:
        print(f"임시 폴더 삭제 오류: {e}")

    print("\n" + "="*50)
    print("🎉 변환 성공!")
    print(f"'{OUTPUT_CT2_PATH}' 폴더가 생성되었습니다.")
    print("="*50)

if __name__ == "__main__":
    convert_model()