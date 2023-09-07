import cv2 as cv
from models.utils import read_image
from functions import loadNetworks, verifyWithRANSAC, saveMatches, visualization
import torch
import os
import numpy as np
from itertools import combinations
import h5py
import exifread
from PIL import Image

if __name__ == '__main__':
    # 数据初始化
    rootPath = './data/'
    pathList = [rootPath+i for i in os.listdir(rootPath)]
    idxs = list(combinations(np.linspace(0, len(pathList)-1, len(pathList)).astype(np.int64), 2))
    # 加载神经网络
    glueType = 'outdoor'
    matchThreshold = 0.3
    device = 'cuda'if torch.cuda.is_available() else 'cpu'
    superpoint, superglue  = loadNetworks(glueType, matchThreshold)
    # 定义匹配数据输出地址
    matchesPath = './output/matches.h5'
    matches = h5py.File(matchesPath, 'w')

    # 穷举法匹配图像对，获取局部特征匹配数据
    for i, j in idxs:
        path0 = pathList[i-1]
        path1 = pathList[j-1]

        image0 = cv.imread(path0, cv.IMREAD_GRAYSCALE)
        image1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
        imageIn0, inpIn0, scales0 = read_image(image0, device, [image0.shape[1], image0.shape[0]], 0, False)
        imageIn1, inpIn1, scales1 = read_image(image1, device, [image1.shape[1], image1.shape[0]], 0, False)
        # SuperPoint输入
        data = {'image0': inpIn0, 'image1': inpIn1}
        # 将图片输入进superpoint特征提取网络
        pred = {}
        pred0 = superpoint({'image': data['image0']})
        pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
        pred1 = superpoint({'image': data['image1']})
        pred = {**pred, **{k + '1': v for k, v in pred1.items()}}
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # 使用SuperGlue进行匹配
        pred = {**pred, **superglue(data)}

        a = 1
        # 将匹配信息写入h5文件中,同时进行几何验证
        saveMatches(matchesPath, pred, i, j, cv.imread(path0), cv.imread(path1))

    # 可视化
    rootOutPath = './output/'
    visualization(matches, rootOutPath)



