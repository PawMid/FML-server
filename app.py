import socket
from serverThread import ServerThread, Aggregator

listenPort = 5000
sendPort = 5001
host = 'localhost'

listenSocket = socket.socket()
listenSocket.bind((host, listenPort))

sendSocket = socket.socket()
sendSocket.bind((host, sendPort))

ag = Aggregator(host)
ag.start()

while True:
    try:
        listenSocket.listen(1)
        sendSocket.listen(1)
        listenConn, addr = listenSocket.accept()
        sendConn, sendAddr = sendSocket.accept()

        print('client connected')
        thread = ServerThread(sendConn, listenConn, addr)
        thread.start()
    except:
        print('Something wrong. Closing socket')
        listenSocket.close()
        sendSocket.close()
        break
