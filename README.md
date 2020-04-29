# Ladder-DenseNet

Source code which reproduces the results from the paper **Efficient Ladder-Style DenseNets for Semantic Segmentation of Large Images** (Ivan Krešo, Josip Krapac, Siniša Šegvić) published in IEEE Transactions on Intelligent Transportation Systems ([link](https://ieeexplore.ieee.org/document/9067100)). Previous work is available on arXiv ([link](https://arxiv.org/abs/1905.05661)).

Demo video on Cityscapes dataset:

<!-- <a href="http://www.youtube.com/watch?feature=player_embedded&v=QrB7Np_8GXY
" target="_blank"><img src="http://img.youtube.com/vi/QrB7Np_8GXY/0.jpg"alt="" width="600" height="400" border="10" /></a> -->


[![http://www.youtube.com/watch?v=QrB7Np_8GXY](http://img.youtube.com/vi/QrB7Np_8GXY/0.jpg)](http://www.youtube.com/watch?v=QrB7Np_8GXY)

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

Pre-trained Cityscapes models are available [here](https://drive.google.com/drive/folders/1VPIspBuXo0YEX4XU1aG3ojnqsdmqcXij?usp=sharing).
* Download and extract the model to `model_path`.


## Evaluation

Evaluate the model on Cityscapes validation subset:

```
python eval.py --model=models/ldn_semseg.py --reader=data/cityscapes/cityscapes_reader.py --dataset=/path/to/cityscapes --weights=/path/to/ldn121_weights_cityscapes_train.pt
```

Enable multiscale inference with `--multiscale-test=1`:

```
python eval.py --model=models/ldn_semseg.py --reader=data/cityscapes/cityscapes_reader.py --dataset=/path/to/cityscapes --weights=/path/to/ldn121_weights_cityscapes_train.pt --multiscale-test=1
```

Save color coded segmentations with `--save-outputs=1`, the images will be saved in the `./outputs` dir.