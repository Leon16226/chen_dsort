import socket

ip_port = ('', 00)


def create_my_socket():
    # 1. socket传递的都是bytes类型的数据，字符串需要转换一下，string.encode()即可
    # 2. accept 和 recv 都是堵塞方法
    # 3. 分为tcp和udp两种

    # 流程
    # 1. 创建socket
    # 2. 连接后发送数据和接受数据，s.sendall和s.recv
    # 3. s.close

    # ipv4 tcp
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(ip_port)

    while True:
        pass

    s.close()
