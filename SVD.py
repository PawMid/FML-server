from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from imgSrc import mainFolder, covid
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA

dirContent = os.listdir(mainFolder + covid)


def findVariances(image):
    pca = PCA()
    img_arr = keras.preprocessing.image.img_to_array(image)[:, :, 0]
    pca.fit(img_arr)
    return np.cumsum(pca.explained_variance_ratio_) * 100


def findElbows(dirContent, treshold=95):
    elbow = []

    for i in dirContent:
        img = Image.open(mainFolder + covid + '\\' + i)
        img = ImageOps.grayscale(img)
        var_cumu = findVariances(img)
        for j in range(len(var_cumu)):
            if var_cumu[j] >= treshold:
                elbow.append(j)
                break
    return elbow


def findMeanElbow(elbows):
    componentSum = 0
    for e in elbows:
        componentSum += e

    return componentSum / len(elbows)


def findMaxComponents(elbows):
    components = 0
    for e in elbows:
        if e > components:
            components = e
    return components


def printVarianceCurve(variances):
    plt.plot(variances)
    plt.show()


image = Image.open(mainFolder + covid + '\\' + dirContent[0])
image = ImageOps.grayscale(image)

elbs = findElbows(dirContent, treshold=98)

meanComponents = findMeanElbow(elbs)
maxComponents = findMaxComponents(elbs)
print(meanComponents)
print(maxComponents)

pca = PCA(n_components=int(meanComponents))
img_arr = keras.preprocessing.image.img_to_array(image)[:, :, 0]
reduced_mean = pca.fit_transform(img_arr)
reduced_mean = pca.inverse_transform(reduced_mean)

printVarianceCurve(findVariances(image))

pca = PCA(n_components=int(maxComponents))
img_arr = keras.preprocessing.image.img_to_array(image)[:, :, 0]
reduced_max = pca.fit_transform(img_arr)
reduced_max = pca.inverse_transform(reduced_max)
printVarianceCurve(findVariances(image))

Image.fromarray(np.hstack((img_arr, reduced_max, reduced_mean))).show()
