import csv
import os
import random
import shutil
from PIL import Image, ImageOps
from termcolor import colored
from progressBar import progressBar
import matplotlib.pyplot as plt
import numpy as np
import collections


class DataLoader:
    def __init__(self, mainPath, classDirs, splitPath, split=0, seed=0, devices=0, proxySplit=0):
        self.__mainPath = mainPath
        self.__images = []
        self.__imageClass = []
        self.__classes = {}  # {0:'covid', 1:'pneumonia', 2:'normal'}
        self.__dataCount = 0
        self.__totalData = 0
        self.__splitPath = splitPath
        self.__split = split
        self.__proxySplit = proxySplit
        self.__seed = seed
        self.__devices = devices
        self.__splitSubDirs = ['train', 'test']
        self.__trainSubDirs = []

        for device in range(devices):
            self.__trainSubDirs.append('device_' + str(device))

        if proxySplit > 0:
            self.__trainSubDirs.append('proxy')

        classCode = 0
        for c in classDirs:
            self.__classes[str(classCode)] = c.replace('\\', '')
            classCode += 1

    def __loadDataset__(self):
        for key in self.__classes:
            curPath = os.path.join(self.__mainPath, self.__classes[key])
            dirContent = os.listdir(curPath)
            for content in dirContent:
                image = (curPath + '\\' + content)
                self.__images.append(image)
                self.__imageClass.append(key)
        self.__dataCount = self.__totalData = len(self.__images)
        print('Data loaded.')

    def __clearSplitDir__(self):
        if os.path.exists(self.__splitPath):
            shutil.rmtree(self.__splitPath)

    def __createSplitDir__(self):
        try:
            os.mkdir(self.__splitPath)
        except OSError:
            print('Failed to create directory.')
        else:
            print('Successfully created directory.')

        try:
            for dir in self.__splitSubDirs:
                path = os.path.join(self.__splitPath, dir)
                os.mkdir(path)
                if dir == 'test':
                    for cl in self.__classes.keys():
                        path = os.path.join(self.__splitPath, dir, self.__classes[cl])
                        os.mkdir(path)
        except OSError:
            print('Failed to create split sub directory.')
            raise OSError
        else:
            print('Successfully created split sub directory.')

        try:
            for dir in self.__trainSubDirs:
                path = os.path.join(self.__splitPath, 'train', dir)
                os.mkdir(path)
                for cl in self.__classes.keys():
                    path = os.path.join(self.__splitPath, 'train', dir, self.__classes[cl])
                    os.mkdir(path)
        except OSError:
            print('Failed to create train sub directory.')
        else:
            print('Successfully created train sub directory.')

    def __getCollection__(self, testAmount):
        lines = []
        random.seed(self.__seed)
        for i in range(testAmount):
            boundary = self.__dataCount
            index = random.randrange(0, boundary)
            self.__dataCount -= 1
            # print('index: ', index, '  ', self.__dataCount)
            line = {'src': self.__images[index], 'class': str(self.__imageClass[index])}
            del self.__imageClass[index]
            del self.__images[index]
            lines.append(line)
        return lines

    def __createCSV__(self, path, lines):
        with open(os.path.join(self.__splitPath, path, 'data.csv'), 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for line in lines:
                writer.writerow([line['src'], line['class']])

    def __createCollection__(self, collection, path, newline=False):
        for element in collection:
            tmpPath = os.path.join(path, self.__classes[element['class']])
            shutil.copy(element['src'], tmpPath)
        progressBar(self.__totalData, self.__dataCount, newline)

    def splitData(self):
        self.__clearSplitDir__()
        self.__createSplitDir__()
        self.__loadDataset__()

        random.seed(self.__seed)

        test = 0

        if self.__split > 0:
            test = int(self.__dataCount * self.__split)
        train = self.__dataCount - test
        proxy = 0
        if self.__proxySplit > 0:
            proxy = int(train * self.__proxySplit)
        devices = train - proxy
        print('Splitting data:')
        devicesToDraw = self.__devices

        testCollection = self.__getCollection__(test)
        testPath = self.__splitPath + '\\test'
        self.__createCollection__(testCollection, testPath)
        self.__createCSV__(testPath, testCollection)
        if self.__proxySplit > 0:
            proxyCollection = self.__getCollection__(proxy)
            proxyPath = self.__splitPath + '\\train\\proxy'
            self.__createCollection__(proxyCollection, proxyPath)
            self.__createCSV__(proxyPath, proxyCollection)

        taken = 0
        previouslyLeft = 0
        maxPool = int(devices / devicesToDraw) + 1
        for i in range(self.__devices):
            pool = self.__dataCount if self.__dataCount <= maxPool else maxPool + previouslyLeft
            entries = random.randint(0, pool)
            previouslyLeft = pool - entries
            taken += entries

            if i == self.__devices - 1:
                entries = self.__dataCount

            collection = self.__getCollection__(entries)
            devicePath = self.__splitPath + '\\train\\' + 'device_' + str(i)
            self.__createCollection__(collection, devicePath, True if i == self.__devices - 1 else False)
            self.__createCSV__(devicePath, collection)

    def __countClasses__(self, device):
        classes = {}
        path = os.path.join(self.__splitPath, device)
        try:
            for cl in self.__classes:
                classes[self.__classes[cl]] = len(os.listdir(os.path.join(path, self.__classes[cl])))
        except OSError:
            print(colored('Path {} does not exist'.format(path), 'red'))

        return classes

    def countClasses(self):
        classes = {-2: self.__countClasses__('test')}
        devices = {-2: 'test'}
        for device in os.listdir(os.path.join(self.__splitPath, 'train')):
            val = self.__countClasses__('train\\'+device)
            if device.find('_') is not -1:
                spl = device.split(sep='_')
                classes[int(spl[1])] = val
                devices[int(spl[1])] = device
            else:
                classes[-1] = val
                devices[-1] = 'proxy'

        classes = collections.OrderedDict(sorted(classes.items()))
        devices = collections.OrderedDict(sorted(devices.items()))

        classesRet = {}

        for cl in self.__classes:
            classesRet[self.__classes[cl]] = []

        for key in classes.keys():
            for k2 in classes[key].keys():
                classesRet[k2].append(classes[key][k2])

        return classesRet, devices

    def plotClasses(self, path='', numOfDevices=10):
        classes, devices = self.countClasses()
        classes = collections.OrderedDict(sorted(classes.items()))
        devices = collections.OrderedDict(sorted(devices.items()))

        if numOfDevices > len(devices) - 2:
            numOfDevices = len(devices) - 2

        indexes = random.sample(range(len(devices) - 2), numOfDevices)
        indexes.append(-2)
        indexes.append(-1)

        devs = []
        cls = {}
        for cl in self.__classes:
            cls[self.__classes[cl]] = []
            for i in indexes:
                cls[self.__classes[cl]].append(classes[self.__classes[cl]][i + 2])

        for i in indexes:
            devs.append(devices[i])

        ind = np.arange(len(devs))
        width = 0.3
        fig, ax = plt.subplots()
        colors = ['red', 'green', 'blue']
        for i in range(len(classes)):
            ax.barh(ind + width * i, list(iter(cls.values()))[i], width, color=colors[i],\
                    label=list(iter(cls.keys()))[i])
        ax.set(yticks=ind + width, yticklabels=devs, ylim=[2 * width - 1, len(devs)])
        ax.legend(loc=4)
        plt.title('Liczność klas w losowych urządzeniach')
        if path is '' or path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()

    def loadData(self, device, grayscale=False):
        data = []

        path = self.__splitPath

        if device == 'test':
            path = os.path.join(path, 'test')
        elif self.__proxySplit > 0 and device == 'proxy':
            path = os.path.join(path, 'train\\proxy')
        elif self.__devices - 1 > device >= 0:
            path = os.path.join(path, 'train\\device_' + str(device))
        else:
            raise ValueError('Invalid device! Available devices: test, ' + \
                             ('proxy, ' if self.__proxySplit > 0 else '') + '0 - ' + str(self.__devices - 1))

        with open(path + '\\data.csv', 'r', newline='') as file:
            reader = csv.reader(file, delimiter=',')

            for line in reader:
                data.append({'image': ImageOps.grayscale(Image.open(line[0])) if grayscale else Image.open(line[0]) \
                                , 'class': line[1]})

        return data