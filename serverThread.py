import threading
import socket
# from convNet1 import convModel
import time
import pickle
import utils
from comunicationCodes import ComCodes

# mainModel = convModel()
# mainModel.loadModelFromFile()
maxDevices = utils.readConfig()
aggregate = False
models = []
mutex = threading.Lock()
counter = 0

class ServerThread(threading.Thread):
    def __init__(self, sendConnection, listenConnection):
        super().__init__()
        self.__sendConnection = sendConnection
        self.__listenConnection = listenConnection
        self.__bufferSize = 1024
        self.__listenerBuffer = []
        self.__listenerMutex = threading.Lock()
        self.__senderBuffer = []
        self.__senderMutex = threading.Lock()

    def __listenerThread(self):
        while True:
            self.__listenerMutex.acquire()
            try:
                received_data = b''
                while str(received_data)[-2] != '.':
                    data = self.__listenConnection.recv(self.__bufferSize)
                    received_data += data
                print(pickle.loads(received_data))
                if received_data != b'':
                    print('self.__listenerBuffer write')
                    self.__listenerBuffer.append(pickle.loads(received_data))
            finally:
                self.__listenerMutex.release()
                time.sleep(0.5)

    def __senderThread(self):
        while True:
            self.__senderMutex.acquire()
            try:
                while not utils.empty(self.__senderBuffer):
                    message = self.__senderBuffer.pop(0)
                    print('sending', message)

                    self.__sendConnection.sendall(pickle.dumps(message))
            finally:
                self.__senderMutex.release()
                time.sleep(0.5)

    def __sendResponse(self, response):
        self.__sendConnection.sendall(pickle.dumps(response))

    def __mainThread(self):

        while True:
            self.__listenerMutex.acquire()
            try:
                while not utils.empty(self.__listenerBuffer):
                    message = self.__listenerBuffer.pop(0)
                    if message == ComCodes.CAN_TRAIN:
                        mutex.acquire()
                        if len(models) < maxDevices:
                            response = {ComCodes.CAN_TRAIN: True}
                        else:
                            response = {ComCodes.CAN_TRAIN: False}
                        self.__sendResponse(response)
            finally:
                self.__listenerMutex.release()
                time.sleep(0.5)

    def run(self):
        # global mainModel, maxDevices, models, aggregate, mutex

        listener = threading.Thread(target=self.__listenerThread)
        sender = threading.Thread(target=self.__senderThread)
        main = threading.Thread(target=self.__mainThread)
        main.start()
        listener.start()
        sender.start()

