from torchvision import transforms
import torch
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import skimage.transform
import math
import os
from data.Hook import actLayerMethod
from data.Hook import maxActivisionValue


# Pics_mapping return list of Image and show all images in set
def imageMapping(path: str, col: int = 3) -> list:
    listDir = os.listdir(path)
    listImage = []
    row = math.ceil(len(listDir) / 3)
    fig = plt.figure(figsize=(12, 12))
    for pcs in range(len(listDir)):
        fig.add_subplot(row, col, pcs + 1)
        listImage.append(Image.open(os.path.join(path, str(listDir[pcs]))))
        plt.imshow(Image.open(os.path.join(path, str(listDir[pcs]))))
        plt.title(label=listDir[pcs], fontsize=5)
        plt.axis('off')
    return listImage


# Regular preprocessing for Image, return torch.tensor ([1, 3, 224, 224])
def preprocessingImage(listImage: list) -> list:
    preprocessingCallable = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        transforms.Compose([transforms.Resize((224, 224))])
    ])
    return [Variable((preprocessingCallable(image).unsqueeze(0)), requires_grad=False) \
            for image in listImage]


# Method create CAM for one pcs and for set of CAM
def camMethod(mapping: torch.Tensor, weight: np.ndarray, index: int, modeFC: bool = True) -> np.ndarray:
    mapping = mapping.cpu().data.numpy()
    _, c, h, w = mapping.shape
    if modeFC == True:
        # Dimension weight[index] = 1x2048, mapping 2048x49 -> output 7x7, index connect with target class
        result = weight[index].dot(mapping.reshape((c, h * w))).reshape(h, w)
    else:
        # Dimension weight[index] = 2048, mapping 2048x49 -> output 7x7
        # Dimension weight[index] = 1024, mapping 1024x192 -> output 14x14
        # Dimension weight[index] = 512, mapping 512x28**2 -> output 28x28
        # Dimension weight[index] = 256, mapping 256x56**2 -> output 56x56
        result = weight.dot(mapping.reshape((c, h * w))).reshape(h, w)

    cam = result - np.min(result)
    cam = cam / np.max(cam)
    return cam


# Cam for set
def camForSet(actFeatures: list, weight: np.ndarray, index: list, modeFC: bool = True) -> np.ndarray:
    return np.array([camMethod(act, weight, idx, modeFC)  # change return type, do we get add dim?
                     for act, idx in zip(actFeatures, index)])


# Visualization for CAM set
def visualizationCAM(path: str, camForPictures: np.ndarray, col: int = 2) -> None:
    listDir = os.listdir(path)
    row = math.ceil(len(listDir) / col)
    fig = plt.figure(figsize=(128, 128))
    display = transforms.Compose([transforms.Resize((224, 224))])

    for pcs in range(len(listDir)):
        fig.add_subplot(row, col, pcs + 1)
        plt.imshow(display(Image.open(os.path.join(path, str(listDir[pcs])))))
        plt.imshow(skimage.transform.resize(camForPictures[pcs],
                                            [224, 224]),
                   alpha=0.7,
                   cmap='jet')

        plt.title(label=listDir[pcs], fontsize=5)
        plt.axis('off')


# Method obtain simulitary and original cam
def camsForLayersRes50(model: torch.nn.Module, preprocessingList: list,
                       listOfWeight: list, startLayer: int = 1) -> list:
    mapping = []
    for i in range(listOfWeight.shape[0]):
        modeFC = False  # mode for last layers FC and layer4's cam. See implementation of actLayerMethod
        weightLayer = np.squeeze(list(listOfWeight[i].parameters())[0] \
                                 .cpu().data.numpy())

        layer = 'layer{}'.format(i + startLayer)  # start from first layer
        idxNumpy = np.empty(len(preprocessingList))  # stub for speed-up, Cam-method have idx for all case

        if listOfWeight[i] == model._modules.get('fc'):
            layer = 'layer4'  # map activisition for fc layer
            idxNumpy = maxActivisionValue(predictionList)
            modeFC = True
        features, predictionList = actLayerMethod(model, preprocessingList,
                                                  layer)
        #         print(i, features[0].cpu().data.numpy().shape, idxNumpy.shape, modeFC ) # control value
        mapping.insert(i, camForSet(features, weightLayer,
                                    idxNumpy, modeFC))

    return mapping
