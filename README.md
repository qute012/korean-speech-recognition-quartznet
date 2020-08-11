<<<<<<< HEAD
An online demo trained with a Mongolian proprietary dataset (WER 8%): [https://chimege.mn/](https://chimege.mn/).

In this repo, following papers are implemented:
* [QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions](https://arxiv.org/abs/1910.10261)
* [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
  * speech recognition as optical character recognition

This repo is partially based on:
* decoder from [SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
* Jasper/QuartzNet blocks from [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

## Training
1. Install PyTorch>=1.3 with conda
2. Install remaining dependencies: `pip install -r requirements.txt`
3. Download the Mongolian Bible dataset: `cd datasets && python dl_mbspeech.py`
4. Pre compute the mel spectrograms: `python preprop_dataset.py --dataset mbspeech`
5. Train: `python train.py --model crnn --max-epochs 50 --dataset mbspeech --lr-warmup-steps 100`
   * logs for the TensorBoard are saved in the folder `logdir`

## Results
During the training, the ground truth and recognized texts are logged into the TensorBoard.
Because the dataset contains only a single person, the predicted texts from the validation set
should be already recognizable after few epochs:

**EXPECTED:**
```
аливаа цус хувцсан дээр үсрэхэд цус үсэрсэн хэсгийг та нар ариун газарт угаагтун
```
**PREDICTED:**
```
аливаа цус хувцсан дээр үсэрхэд цус усарсан хэсхийг та нар ариун газарт угаагтун
```

For fun, you can also generate an audio with a Mongolian TTS and try to recognize it.
The following code generates an audio with the
[TTS of the Mongolian National University](http://172.104.34.197/nlp-web-demo/)
and does speech recognition on that generated audio:
```
# generate audio for 'Миний төрсөн нутаг Монголын сайхан орон'
wget -O test.wav "http://172.104.34.197/nlp-web-demo/tts?voice=1&text=Миний төрсөн нутаг Монголын сайхан орон."
# speech recognition on that TTS generated audio
python transcribe.py --checkpoint=logdir/mbspeech_crnn_sgd_wd1e-05/epoch-0050.pth --model=crnn test.wav
# will output: 'миний төрсөн нут мөнголын сайхан оөрулн'
```

It is also possible to use a KenLM binary model. First download it from
[tugstugi/mongolian-nlp](https://github.com/tugstugi/mongolian-nlp#mongolian-language-model).
After that, install [parlance/ctcdecode](https://github.com/parlance/ctcdecode). Now you can transcribe with the language model:
```
python transcribe.py --checkpoint=path/to/checkpoint --lm=mn_5gram.binary --alpha=0.3 test.wav
```

## Contribute
If you are Mongolian and want to help us, please record your voice on [Common Voice](https://voice.mozilla.org/mn/speak).
=======
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
>>>>>>> b7c9f6289802374e46a6085b8d031500c2f2d4ad
