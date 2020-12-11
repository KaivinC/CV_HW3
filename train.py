# -*- coding: utf-8 -*-

import os
import cv2
import json
import random
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor,launch
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from itertools import groupby
from pycocotools import mask as maskutil
from detectron2 import model_zoo
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.build import build_detection_train_loader,build_detection_test_loader
import copy
import torch
import warnings
import argparse
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser("Detectron2")
    parser.add_argument("--gpu_num", type=int, default=1,
                        help="The number of gpu you will use")
    parser.add_argument("--batch_size_per_gpu", type=int, default=5,
                        help="The number of images per batch each GPU")
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--num_iters", type=int, default=100000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    return args

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    image, transforms = T.apply_transform_gens([
        T.RandomFlip(prob=0.50, horizontal=True, vertical=False),
        T.RandomApply(tfm_or_aug=T.RandomBrightness(intensity_min=0.7, intensity_max=1.1),
                      prob=0.40),
        T.RandomApply(tfm_or_aug=T.RandomSaturation(intensity_min=0.7, intensity_max=1.1), 
                      prob=0.40)
    ], image)
    
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class Trainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)
        
def main(args):
    setup_logger()
    train_path = "./datas/train_images/"
    json_file = "./datas/pascal_train.json"
    # convert COCO format to Detectron2 format
    register_coco_instances("VOC_dataset", {}, json_file, train_path)
    dataset_dicts = load_coco_json(json_file, train_path, "VOC_dataset")

    VOC_metadata = MetadataCatalog.get("VOC_dataset")


    os.makedirs('train_results', exist_ok=True)
    # ============ train ===========
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    if(args.resume):
        cfg.MODEL.WEIGHTS = args.resume
    cfg.DATASETS.TRAIN = ("VOC_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = args.gpu_num*args.batch_size_per_gpu
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.num_iters
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    tr = Trainer(cfg)
    tr.resume_or_load(resume=False)
    tr.train()

    # ============= training results ===========
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")) 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    cfg.DATASETS.TEST = ("VOC_dataset", )
    predictor = DefaultPredictor(cfg)
    json_file = "./datas/pascal_train.json"
    cocoGt = COCO(json_file)

    coco_dt_train = []
    for imgid in cocoGt.imgs:
        filename = cocoGt.loadImgs(ids=imgid)[0]['file_name']
        print('predicting ' + filename)
        im = cv2.imread(train_path + filename)  # load image
        outputs = predictor(im)  # run inference of your model

        output_path = os.path.join('train_results', filename)
        v = Visualizer(im[:, :, ::-1],
                       metadata=VOC_metadata,
                       scale=3,
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(output_path, v.get_image()[:, :, ::-1])

        anno = outputs["instances"].to("cpu").get_fields()
        masks = anno['pred_masks'].numpy()
        categories = anno['pred_classes'].numpy()
        scores = anno['scores'].numpy()

        n_instances = len(scores)
        if len(categories) > 0:  # If any objects are detected in this image
            for i in range(n_instances):  # Loop all instances
                # save information of the instance in a dictionary then append on coco_dt list
                pred = {}
                pred['image_id'] = imgid  # this imgid must be same as the key of test.json
                pred['category_id'] = int(categories[i]) + 1
                # save binary mask to RLE, e.g. 512x512 -> rle
                pred['segmentation'] = binary_mask_to_rle(masks[i, :, :])
                pred['score'] = float(scores[i])
                coco_dt_train.append(pred)

    with open("train_result.json", "w") as f:
        json.dump(coco_dt_train, f)

    cocoDt = cocoGt.loadRes("train_result.json")

    imgIds = sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    args = get_args()
    launch(main,num_gpus_per_machine=args.gpu_num, dist_url="auto",args=(args,))