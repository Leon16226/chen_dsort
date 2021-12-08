import socket
import threading
import json


def my_response(is_suceess, msg, data, timestamp):
    response = {
        'is_success': is_suceess,
        'msg': msg,
        'data': data,
        'requestTimestamp': timestamp,
        'responseTimestamp': timestamp
    }

    my_json = json.dumps(response, ensure_ascii=False)
    return my_json


class My_Socket():

    def __init__(self):
        self.running = True
        self.sk = None
        self.links = []

    def terminate(self,):
        self.running = False

    def link_handler(self, link, client, control={}, path={}, detect_live=[False], ip_states={}):

        print('服务端接受来自[%s:%s]的请求...' % (client[0], client[1]))

        while self.running:
            client_date = link.recv(1024).decode('utf-8')

            print('[%s:%s]发来一条消息...' % (client[0], client[1]))
            if len(client_date) == 0:
                print('此消息为空，返回...')
                continue

            user_dic = json.loads(client_date)

            # method
            method = user_dic['method']
            requestTimestamp = user_dic['requestTimestamp']

            if method == 'control':
                # 1. 控制检测程序-----------------------------------------------------------------------------------------
                print("接收到控制消息...")
                data = user_dic['data']
                command = data['command']

                path['eventUploadPath'] = data['eventUploadPath']
                path['trafficUploadPath'] = data['trafficUploadPath']
                path['configsPath'] = data['configsPath']
                path['carNoUploadPath'] = data['carNoUploadPath']

                print(command)

                # 停止检测
                if command == 'stop':
                    control['stop'] = True

                elif command == 'start' or command == 'restart':
                    my_json = my_response(True, '成功', None, requestTimestamp)
                    print(my_json)
                    n = link.sendall(my_json.encode())
                    if n is None:
                        print('返回信息成功...')
                    control['restart'] = True
                break
            elif method == 'service_status':
                # 2. 返回服务器检测状态-----------------------------------------------------------------------------------------
                print("接收到服务器检测状态消息...")

                # 0停止，1运行中，2故障
                status = 1 if detect_live[0] else 0
                data = {
                    "status": status,
                    "remark": ""
                }
                my_json = my_response(True, '成功', data, requestTimestamp)
                print(my_json)
                link.sendall(my_json.encode())
                break
            elif method == 'camera_status':
                # 3. 返回点位检测状态-------------------------------------------------------------------------------------
                print("接受到点位检测状态消息...")

                # 0未检测，1检测中
                data = user_dic['data']
                ip = data['ip']
                if ip in ip_states.keys():
                    status = 1 if ip_states.get(ip) else 0
                    remark = ''
                else:
                    status = 0
                    remark = "当前没有检测这个点位..."
                data = {
                    "status": status,
                    "remark": remark,
                }
                my_json = my_response(True, '成功', data, requestTimestamp)
                print(my_json)
                link.sendall(my_json.encode())
                break

        print('socket close...')
        link.close()

    def create_socket_service(self, ip='192.168.1.149', port=4000,
                              control={}, path={}, detect_live=[False], ip_states={}):

        ip_port = (ip, port)
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sk.bind(ip_port)
        self.sk.listen(5)

        print('启动socket服务，等待客户端连接...')

        while self.running:
            print('accepting...')
            conn, address = self.sk.accept()

            t = threading.Thread(target=self.link_handler, args=(conn, address, control, path, detect_live, ip_states))
            t.start()
            self.links.append(conn)

        self.sk.close()

    def to_close(self):
        print('关闭所有socket连接...')
        self.sk.close()



