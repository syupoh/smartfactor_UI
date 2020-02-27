import socket
import pickle
from .classifier import Classifier
import numpy as np
from codes.config import SERVER_HOST, SERVER_PORT, PACKET_SIZE
from ..socket_funcs import receive, send


__all__ = ['BasicServer', 'Server']


class BasicServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __del__(self):
        self.socket.close()

    def sendObj(self, conn, obj):
        data = pickle.dumps(obj)
        send(conn, data)

    def receive(self, conn):
        data = receive(conn, PACKET_SIZE)
        obj = pickle.loads(data)
        return obj

    def onReceived(self, conn, obj):
        raise NotImplementedError()
        self.sendObj(conn, obj)
        conn.close()

    def main(self):
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print('server ready')
        while True:
            conn, addr = self.socket.accept()
            print('connection estabilished from:', addr)

            obj = self.receive(conn)
            print('Data received')
            self.onReceived(conn, obj)


class Server(BasicServer):
    def __init__(self):
        super(Server, self).__init__('0.0.0.0', SERVER_PORT)
        self.classifier = Classifier()

    def onReceived(self, conn, obj):
        print('Received np array:', obj.shape)
        result = self.classifier.classify(obj)
        # result = np.ones(obj.shape[0])
        self.sendObj(conn, result)
