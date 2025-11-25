# handDoc - Speech-AI
FastAPI 기반으로 음성 파일을 받아 파인튜닝된 Whisper 모델을 통해 텍스트 결과를 반환합니다. 

## 🛠️ 기술 스택
- 데이터셋 : [구음장애 음성 인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
- 모델 : [OpenAI whisper-small](https://huggingface.co/openai/whisper-small) 모델
- 학습 설정 : PEFT (Parameter-Efficient Fine-Tuning) via QLoRA 

## 🚀 실행 방법

```bash
# 가상환경 생성 및 활성화
# Windows 
python -m venv venv 
source venv/Scripts/activate

# Mac / Linux
python3 -m venv venv 
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
