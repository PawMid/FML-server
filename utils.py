import io
import zlib
import numpy as np


def compress_nparr(nparr):
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed


def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))


def empty(arr):
    return len(arr) == 0


def readConfig(path='..\\config.txt'):
    nDevices = None
    with open(path, 'r') as config:
        lines = config.readlines()
        for line in lines:
            line = line.replace(' ', '')
            line = line.split('=')
            if line[0] == 'nDevices':
                nDevices = int(line[1])
    return nDevices


def readNetTypes(path='..\\config.txt'):
    return __readProperty('nets', path).split(',')


def getClientsNumber(path='..\\config.txt'):
    return int(__readProperty('nDevices', path))


def getTesterPort(path='..\\config.txt'):
    return int(__readProperty('testerPort', path))


def getServerPort(path='..\\config.txt'):
    return int(__readProperty('serverPort', path))


def __readProperty(prop, path='..\\config.txt'):
    with open(path, 'r') as config:
        lines = config.readlines()
        for line in lines:
            line = line.replace(' ', '')
            line = line.split('=')
            if line[0] == prop:
                return line[1]
