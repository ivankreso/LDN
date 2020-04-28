# Ladder-DenseNet

## Requirements
* Python (>= 3.7)
```
pip install torch pillow numpy tqdm
```

## Preparation

### Download Cityscapes

Download and unzip following dataset files from https://www.cityscapes-dataset.com/downloads/:
* leftImg8bit_trainvaltest.zip
* gtFine_trainvaltest.zip

Place both `leftImg8bit` and `gtFine` dirs into the same dataset root dir `dataset_path`.

### Download pre-trained weights

Pre-trained Cityscapes models are available [here](https://drive.google.com/drive/folders/1VPIspBuXo0YEX4XU1aG3ojnqsdmqcXij?usp=sharing)
* Download and extract the model to `model_path`.


## Evaluation

Evaluate the model on Cityscapes validation subset:

```
python eval.py --model=models/ldn_semseg.py --reader=data/cityscapes/cityscapes_reader.py --dataset=/path/to/cityscapes --weights=/path/to/ldn121_weights_cityscapes_train.pt
```

```
python eval.py --model=models/ldn_semseg.py --reader=data/cityscapes/cityscapes_reader.py --dataset=/path/to/cityscapes --weights=/path/to/ldn121_weights_cityscapes_train.pt --multiscale-test=1
```