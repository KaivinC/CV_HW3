# HW3

## Abstract

In this work, I use Detectron2(MaskRCNN+ ResNext101 + FPN ) to train my model

MaskRCNN
[Paper](https://arxiv.org/abs/1703.06870)

Detectron2
[Github](https://github.com/facebookresearch/detectron2)

## Reproducing Submission

To reproduct my submission without retrainig, do the following steps

1. [Installation](#installation)
2. [Download Official Image](#download-official-image)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation

```bash
python3 -m pip3 install -r requirements.txt
```

## Dataset Prepare

### Download Official Image

Download the data from [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK)

### Prepare Images

After downloading, the data directory is structured as

```
datas
    +- train_images //1349 images
    +- test_images  //100 images
    +- pascal_train.json
    +- test.json
```



## Training

My final submission is use Detectron2(MaskRCNN+ ResNext101 + FPN )

Run `train.py` to train.

```bash
python3 train.py --gpu_num = {number of gpu} \
--batch_size_per_gpu = {batch size per gpu} \
--resume = {pretrain model path if have}

```

The expected training times are

Model | GPUs | Image size | Training iter | Training Time
------------ | ------------- | ------------- | ------------- | -------------
efficientDet | 2x 2080Ti | 800*1000 | 100000 | 24 hours

## Pretrained models

You can download pretrained model that used for my submission from [link](https://drive.google.com/drive/folders/1wJ-HesZKWnCBBgzkJ2e7aU0mg-uIsLET?usp=sharing).

Unzip them into results then you can see following structure

```bash
+- imagenet_pretrain_model
    +- X-101-32x8d.pkl
+- final_model.pth
```

## Inference

If trained weights are prepared, you can create the result file which named result.json by run below command

```bash
$python3 test.py --resume={trained model path}
```

## Visualize the result

It will visualize result automatically in the train_result and test_result.

## Make Submission

Click [here](https://drive.google.com/drive/folders/1VhuHvCyz2CH4yzDreyVTwhZiOFbQB09B) to submission the json file!!

## Citation

```

@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}

```