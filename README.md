# image-captioning-Pytorch

## How to run my code
1. Install Python dependency
```bash
# Install requirements
pip install -r requirements.txt
# Install cocoapi
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install
```
2. Coco 2014, Coco 2017 데이터를 다운로드합니다. 다음과 같은 폴더 구조로 저장했습니다.
- `coco2014`, `coco2017`: [https://cocodataset.org](https://cocodataset.org/)
- `coco2014/caption_datasets`: [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
```text
.
├── coco2014
│   ├── train2014
│   │   └── ...
│   ├── val2014
│   │   └── ...
│   ├── caption_datasets
│				└── dataset_coco.json
│   └── annotations
│				├── captions_train2014.json
│		    ├── captions_test2014.json               # Run split_coco2014.py
│		    └── captions_val2014.json                # Run split_coco2014.py
├── coco2017
│   ├── train2017
│   │   └── ...
│   ├── val2017
│   │   └── ...
│   └── annotations
│				├── captions_train2017.json
│				├── captions_train2017_clip_top4.json     # Run clip.py
│				├── captions_train2017_clip_top3.json     # Run clip.py
│		    └── captions_val2017.json
```
3. wordmap.json을 생성합니다.
```bash
python -m preprocess.create_wordmap
```
4. 필요하다면 top-4 caption, top-3 caption 데이터를 만듭니다.
```bash
python -m preprocess.clip
```
5. Coco 2014 데이터를 split 합니다.
```bash
python -m preprocess.split_coco2014
```
6. 모델을 학습합니다. 다양한 학습을 실험하기 위해서 몇몇 파이썬 파일을 수정해야 합니다. 제 코드는 최종 모델을 학습시킵니다.
- learning rate: `lightningmodule.py`
- train / valid / test dataset: `datamodule.py`
- Train Encoder or Decoder: `train.py`
```bash
python train.py
```
7. eval.py로 모델을 평가하고, caption.py로 inference를 수행할 수 있습니다.
```bash
python eval.py
python caption.py \
	--img <path_to_image> \
	--model <path_to_model> \
	--beam_size <beam_size>
```
