import os
from PIL import Image, ImageOps


def openImg(path, imgNum, grayscale=False):
    dirContent = os.listdir(path)
    img = Image.open(path + '\\' + dirContent[imgNum])
    if grayscale:
        return ImageOps.grayscale(img)
    return img
