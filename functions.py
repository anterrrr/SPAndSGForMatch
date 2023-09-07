import numpy as np
import cv2 as cv
import torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
import h5py
import exifread
from PIL import Image


def loadNetworks(glueType, matchThreshold):
    """
    函数解释：用于加载SuperPoint和SuperGlue
    :param glueType: SuperGlue的预训练模型参数类型 “outdoor” or "indoor"
    :param matchThreshold: sinkhorn算法判定是否匹配的阈值大小
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nms_radius = 4
    keypoint_threshold = 0.005
    max_keypoints = 2048
    configPoint = {'nms_radius': nms_radius,
                   'keypoint_threshold': keypoint_threshold,
                   'max_keypoints': max_keypoints
                   }

    weights = glueType
    sinkhorn_iterations = 20
    match_threshold = matchThreshold
    configGlue = {'weights': weights,
                  'sinkhorn_iterations': sinkhorn_iterations,
                  'match_threshold': match_threshold,
                  }
    superpoint = SuperPoint(configPoint).to(device)
    superpoint.load_state_dict(torch.load('./models/weights/superpoint_v1.pth'))
    superglue = SuperGlue(configGlue).to(device)
    superglue.load_state_dict(torch.load('./models/weights/superglue_'+glueType+'.pth'))
    # return {'superpoint': superpoint, 'superglue': superglue}
    return superpoint, superglue


def verifyWithRANSAC(pointsA, pointsB):
    """
    函数解释：使用八点发和RANSAC对SuperGlue的预测结果进行几何验证
    :param pointsA: A图像上的特征点坐标
    :param pointsB: 对应的B图像上的特征点坐标
    :return: 通过验证的内点坐标
    """
    # 将输入点转换为numpy数组
    pointsA = np.array(pointsA, dtype=np.float32)
    pointsB = np.array(pointsB, dtype=np.float32)
    # 计算基本矩阵 8-points algorithm with RANSAC
    if len(pointsA) < 8 or len(pointsB) < 8:
        raise ValueError("At least 8 point pairs are required for the 8-point algorithm.")

    fundamentalMatrix, inliersMask = cv.findFundamentalMat(pointsA, pointsB, method=cv.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99)

    # 使用掩码筛选内点
    inliersA = pointsA[inliersMask.ravel() == 1]
    inliersB = pointsB[inliersMask.ravel() == 1]
    return inliersA, inliersB

def saveIntrinsic(intrinsicPath, pathList):
    """
    函数解释：从图片EXIF中读取相机的内参数并存入.h5文件中
    :param intrinsicPath:.h文件
    :param pathList: 图像路径
    :return:
    """
    intrinsic = h5py.File(intrinsicPath, 'w')
    for i in range(len(pathList)):
        image = open(pathList[i], 'rb')
        f = float(exifread.process_file(image)['EXIF FocalLength'].values[0])
        fx, fy = f, f
        image = Image.open(pathList[i])
        cx = image.width / 2
        cy = image.height / 2
        IntrinsicMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        with h5py.File(intrinsicPath, 'a', libver='latest') as fa:
            grp = fa.create_group('image' + f'{i + 1}')
            grp.create_dataset('intrinsic', data=IntrinsicMatrix)


def saveMatches(matchesPath, pred, i, j, image0, image1):
    """
    函数解释：将匹配信息写入.h5文件中
    :param matchesPath:
    :param pred: superGlen预测结果
    :param i: image0索引
    :param j: image1索引
    :param image0: image0的图像数据  三维数组
    :param image1: image1的图像数据  三维数组
    :return: None
    """
    # pred:superGlue输出
    # 创建.h5文件用于存放匹配数据
    with h5py.File(matchesPath, 'a', libver='latest') as f:
        grp = f.create_group('image' + f'{i}' + '_image' + f'{j}')

        # 获取匹配索引
        idx = pred['matches0'].cpu().numpy()
        idx0 = [m for m in range(idx.shape[1]) if idx[0, m] != -1]
        idx1 = [idx[0, n] for n in idx0]
        # 获取对应坐标
        kpts0 = pred['kpts0'][0].cpu().numpy()
        kpts0 = kpts0[idx0]
        kpts1 = pred['kpts1'][0].cpu().numpy()
        kpts1 = kpts1[idx1]

        # 使用八点法计算基础矩阵进行几何验证
        kpts0, kpts1 = verifyWithRANSAC(kpts0, kpts1)
        grp.create_dataset('kpts', data=np.concatenate((kpts0, kpts1), axis=1)).astype(np.int32)

        # 存入图像数据
        grp.create_dataset('image0', data=image0)
        grp.create_dataset('image1', data=image1)

        # # 写入匹配得分
        # scores = pred['matching_scores0'].cpu().detach().numpy()
        # scores = [p for p in scores.T if p > 0.5]
        # grp.create_dataset('scores', data=scores)


def visualization(matches, rootPath):
    """
    函数解释：将匹配结果在一张图像上可视化
    :param matches: .h5文件中的匹配信息(image, kpts)
    :param rootPath: 最终可视化的写入根目录
    :return: None
    """
    for i in matches:
        image0 = matches[i]['image0'].__array__()
        image1 = matches[i]['image1'].__array__()
        kpts0 = matches[i]['kpts'].__array__()[:, 0:2]
        kpts1 = matches[i]['kpts'].__array__()[:, 2:4]
        outPath = rootPath+i+'.jpg'


        # 新建画布
        image = np.zeros([max(image0.shape[0], image1.shape[0]), image0.shape[1] + image1.shape[1], 3])
        image[0:image0.shape[0], 0:image0.shape[1], :] = image0
        image[0:image1.shape[0], image0.shape[1]:image0.shape[1] + image1.shape[1], :] = image1
        increment = image0.shape[1]
        kpts1 = np.array([[m+increment, n] for m, n in kpts1])

        # 画点
        for i in kpts0:
            image[int(i[1]), int(i[0])] = 255
            cv.circle(image, (int(i[0]), int(i[1])), 2, color=[255, 255, 0], thickness=-1)
        for i in kpts1:
            image[int(i[1]), int(i[0])] = 255
            cv.circle(image, (int(i[0]), int(i[1])), 2, color=[255, 255, 0], thickness=-1)

        # 连线
        for i in range(len(kpts0)):
            startPoint = tuple(np.round(kpts0[i]).astype(np.int64))
            endPoint = tuple(np.round(kpts1[i]).astype(np.int64))
            cv.line(image, startPoint, endPoint, (255, 0, 0), 1)

        # write
        cv.imwrite(outPath, image)



