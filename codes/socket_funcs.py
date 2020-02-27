import socket


def send(soc, data):
    """

    :param socket.socket soc:
    :param bytes data:

    :return:
    """
    length = len(data)

    soc.send(str(length).encode('ascii'))

    ok = soc.recv(1024)

    soc.send(data)


def receive(soc, packet_size):
    """

    :param socket.socket soc:
    :param int packet_size:
    :return:
    """
    data = b''
    len_bytes = soc.recv(packet_size)
    length = int(len_bytes.decode('ascii'))

    soc.send(b'ok')

    while True:
        recv = soc.recv(packet_size)
        data += recv

        if len(data) >= length:
            break
    return data
