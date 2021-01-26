import threading
import sys
from convNet1 import convModel
import time
import pickle
import utils
import zlib
import socket
import numpy as np
from comunicationCodes import ComCodes
import imgSrc

structure = 'vgg'
mainModel = convModel('proxy')
mainModel.loadModelFromFile(structure)
maxDevices = utils.readConfig() - 1
preAccuracy = None
postAccuracy = None

models = {}
mutex = threading.Lock()



class ServerThread(threading.Thread):
    def __init__(self, sendConnection, listenConnection, addr):
        super().__init__()
        self.__address = addr
        print(addr)
        self.__sendConnection = sendConnection
        self.__listenConnection = listenConnection
        self.__bufferSize = 1024
        self.__listenerBuffer = []
        self.__listenerMutex = threading.Lock()
        self.__senderBuffer = []
        self.__senderMutex = threading.Lock()
        self.__closeConnection = False
        self.setName(addr)

    def __listenerThread(self):
        while True:

            try:
                received_data = b''
                size = pickle.loads(self.__listenConnection.recv(self.__bufferSize))

                while sys.getsizeof(received_data) < size:
                    data = self.__listenConnection.recv(self.__bufferSize)
                    received_data += data
                self.__listenerMutex.acquire()
                if received_data != b'':
                    self.__listenerBuffer.append(pickle.loads(zlib.decompress(received_data)))
                    self.__listenerMutex.release()
            except:
                self.__closeConnection = True
                break
            finally:
                time.sleep(1)

    def __senderThread(self):
        while True:
            if self.__closeConnection:
                break
            self.__senderMutex.acquire()
            try:
                while not utils.empty(self.__senderBuffer):
                    message = self.__senderBuffer.pop(0)
                    # print('sending', message)

                    self.__sendConnection.sendall(pickle.dumps(message))
            finally:
                self.__senderMutex.release()
                time.sleep(0.5)

    def __sendResponse(self, response):
        resp = zlib.compress(pickle.dumps(response), 9)
        size = sys.getsizeof(resp)
        # print('sending', response)
        self.__sendConnection.sendall(pickle.dumps(size))
        time.sleep(1)
        self.__sendConnection.sendall(resp)

    def __mainThread(self):
        while True:
            if self.__closeConnection:
                break
            self.__listenerMutex.acquire()
            try:
                if not utils.empty(self.__listenerBuffer):
                    # mutex.acquire()
                    message = self.__listenerBuffer.pop(0)
                    messageCode = message[0]
                    if messageCode == ComCodes.CAN_TRAIN:
                        if len(models) < maxDevices:
                            mutex.acquire()
                            models[self.__address] = []
                            mutex.release()
                            response = (ComCodes.CAN_TRAIN, True)
                        else:
                            response = (ComCodes.CAN_TRAIN, False)
                        # response = (ComCodes.CAN_TRAIN, True)
                        self.__sendResponse(response)
                    elif messageCode == ComCodes.GET_STRUCTURE:
                        self.__sendResponse((ComCodes.GET_STRUCTURE, structure))
                    elif messageCode == ComCodes.GET_WEIGHTS:
                        response = (ComCodes.GET_WEIGHTS, mainModel.getTrainableWeights())
                        self.__sendResponse(response)
                    elif messageCode == ComCodes.POST_WEIGHTS:
                        try:
                            mutex.acquire()
                            models[self.__address] = message[1]
                        finally:
                            mutex.release()
            finally:
                # mutex.release()
                self.__listenerMutex.release()
                time.sleep(0.5)

    def run(self):
        # global mainModel, maxDevices, models, aggregate, mutex

        listener = threading.Thread(target=self.__listenerThread)
        listener.setName(str(self.__address) + "-listener")

        # sender = threading.Thread(target=self.__senderThread)

        main = threading.Thread(target=self.__mainThread)
        main.setName(str(self.__address) + "-main")

        main.start()
        listener.start()
        # sender.start()


class Aggregator(threading.Thread):
    """
    Class responsible for federated models weight aggregation, sending results to controller
    """
    def __init__(self, host):
        super().__init__()
        self.setName('Aggregator')
        self.__aggregated = None

        self.__host = host
        self.sendPort = utils.getServerPort() + 1
        self.listenPort = utils.getServerPort()
        self.__bufferSize = 1024

        self.__listenSocket = socket.socket()
        self.__listenSocket.bind((host, self.listenPort))
        self.__listenConn = None

        self.__sendSocket = socket.socket()
        self.__sendSocket.bind((host, self.sendPort))
        self.__sendConn = None

    def __aggregate(self, items):
        """
        Method for aggregating data. Performs mean aggregation.
        :param items: array of multiple models trainable weights.
        """
        numOfModels = len(items)
        layers = len(items[0])
        aggregatedWeights = []

        for layer in range(layers):
            temp = items[0][layer].numpy()
            for model in range(1, numOfModels):
                temp = temp + items[model][layer].numpy()
                temp = (temp / numOfModels).astype(np.float16)
            aggregatedWeights.append(temp)
        aggregatedWeights = np.array(aggregatedWeights)
        self.__aggregated = aggregatedWeights

    def dictToArr(self, dict):
        """

        :param dict: dictionary with weights
        :return: array of weights
        """
        weights = []
        for key in dict.keys():
            weights.append(dict[key])
        return weights

    def __isReadyToAggregate(self):
        """

        :return: True if weights are ready to aggregate otherwise False.
        """
        if maxDevices <= len(models):
            for key in models.keys():
                if len(models[key]) <= 0:
                    return False
            return True
        else:
            return False

    def send(self, message):
        # print('sending', message)
        resp = zlib.compress(pickle.dumps(message), 4)
        size = sys.getsizeof(resp)

        self.__sendConn.sendall(pickle.dumps(size))
        time.sleep(1)
        self.__sendConn.sendall(resp)

    def __listenerThread(self):
        global models, preAccuracy, mainModel, structure
        while True:

            received_data = b''
            size = pickle.loads(self.__listenConn.recv(self.__bufferSize))
            print('Device', self.getName(),'receiving data of size', size)
            while sys.getsizeof(received_data) < size:
                data = self.__listenConn.recv(self.__bufferSize)
                received_data += data
            if received_data != b'':
                response = (pickle.loads(zlib.decompress(received_data)))
                print(response)
                if response[0] == ComCodes.LOAD_MODEL:
                    modelType = response[1]
                    mutex.acquire()
                    structure = modelType
                    mainModel.loadModelFromFile(modelType)
                    models = {}
                    preAccuracy = mainModel.getAccuracy()[0]
                    mutex.release()
                    # self.send([ComCodes.LOAD_MODEL, True])
                    print('sending acc')
                    self.send([ComCodes.POST_ACCURACY, [preAccuracy, '-']])
                elif response[0] == ComCodes.GET_STRUCTURE:
                    mutex.acquire()
                    self.send([ComCodes.GET_STRUCTURE, structure])
                    mutex.release()
                elif response[0] == ComCodes.GET_WEIGHTS:
                    mutex.acquire()
                    self.send([ComCodes.GET_WEIGHTS, mainModel.getTrainableWeights(), preAccuracy])
                    mutex.release()

    def run(self):
        global models, preAccuracy, postAccuracy

        print('Waiting for controller.')
        self.__listenSocket.listen(1)
        self.__sendSocket.listen(1)
        self.__listenConn, addr = self.__listenSocket.accept()
        self.__sendConn, sendAddr = self.__sendSocket.accept()
        print('Controller connected.')

        listener = threading.Thread(target=self.__listenerThread, name='aggregator-listener')
        listener.start()

        while True:
            if preAccuracy is None:
                mutex.acquire()
                preAccuracy = mainModel.getAccuracy()[1]
                mutex.release()
                self.send([ComCodes.LOAD_MODEL, True])
                time.sleep(0.8)
                self.send([ComCodes.POST_ACCURACY, [preAccuracy, '-']])
            if self.__isReadyToAggregate():
                self.__aggregate(self.dictToArr(models))
            if self.__aggregated is not None:
                try:
                    mutex.acquire()
                    mainModel.setTrainableWeights(self.__aggregated)

                    mainModel.getConfusionMatrix(imgSrc.results, 'post_train')
                    postAccuracy = mainModel.getAccuracy()[1]

                    models = {}
                    self.__aggregated = None

                    print(preAccuracy, postAccuracy)
                    self.send([ComCodes.POST_ACCURACY, [preAccuracy, postAccuracy]])
                    preAccuracy = postAccuracy
                    postAccuracy = None
                finally:
                    mutex.release()
            time.sleep(1)
