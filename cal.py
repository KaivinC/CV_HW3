import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def main():
    images = []
    cnt = 0
    for file in tqdm(glob.glob("/home/kschen/CV_hw3/CVHW3/datas/train_images/*.jpg")):
        img = cv2.imread(file)
        images.append(list(img.shape))
        cnt += 1
    for file in tqdm(glob.glob("/home/kschen/CV_hw3/CVHW3/datas/test_images/*.jpg")):
        img = cv2.imread(file)
        images.append(list(img.shape))
        cnt += 1
    result = [0,0,0]
    for ele in images:
        result = [sum(x) for x in zip(result,ele)]
    result = [x/cnt for x in result]
    print(result)
if __name__ == '__main__':
    main()