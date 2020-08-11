# quartznet-ko
Jasper 양자화 모델인 Quartznet에 Ai-Hub를 학습한 한국어 음성인식입니다.

본 프로젝트는 https://github.com/tugstugi/mongolian-speech-recognition에서 확장되었습니다.

# Preparation
/datasets/meta/aihub 경로에 char2idx(dict), idx2char(list)를 피클로 저장

/datasets/meta/aihub 경로에 train data와 test data를 아래와 같은 포맷으로 저장

tuple 자료형으로 음성파일 경로와 자막을 담고있는 리스트를 피클로 저장 e.g. [(음성파일 경로, 자막)]

# Training
```python3 train.py --dataset aihub --model quartznet5x5 --max-epochs 500 --train-batch-size 16 --valid-batch-size 16 --lr 3e-4 --weight-decay 0 --lr-policy none```

# Performance
성능은 한국어 위키피디아 데이터를 KenLM을 사용하여 4grams rescoring을 하였을 때, 가장 높은 것으로 확인하였습니다.

|Model|CER|WER|
|------|---|---|
|Only AM|16%|46%|
|AM+4grams|8%|16%|
