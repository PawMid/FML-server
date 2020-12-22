from kmeans_pytorch import kmeans
from torchvision import transforms
import numpy as np

from openIMG import openImg
from imgSrc import mainFolder, covid
import matplotlib.pyplot as plt

mean = (0, 0, 0)
std = (255, 255, 255)

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

img = openImg(mainFolder+covid, 0)

img_arr = np.array(img)
transformed = transform(img)
print(transformed)
plt.imshow(transformed[0, :, :], cmap='gray')
plt.show()
print((img_arr.shape))




