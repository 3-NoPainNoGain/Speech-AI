import librosa
import io
import os
from whisper_ctranslate2 import WhisperModel 
from transformers import WhisperProcessor

MODEL_PATH = "./models_ct2"
DEVICE = "cpu"

model = None
processor = None

def load_model():
    global model, processor
    
    if model is None:
        print("🚀 CTranslate2 Whisper 모델을 로드하는 중...")
        model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type="int8")
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        
        print("✅ CTranslate2 모델 로드 완료.")

def transcribe_audio_file(audio_bytes:bytes) -> str:
    if model is None or processor is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    
    audio_steam = io.BytesIO(audio_bytes)
    speech_array, _ = librosa.load(audio_steam, sr=16000, mono=True)
    
    print("Running CTranslate2 inference...")
    
    segments, info = model.transcribe(
        speech_array,
        language="ko", 
        task="transcribe",
        without_timestamps=True,
    )

    transcription = "".join(segment.text for segment in segments)
    
    return transcription.strip()