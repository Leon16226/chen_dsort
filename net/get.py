import cv2
import base64
import time
import datetime
import json
import requests
from .event import Event
from utils.myjson import read_json



def get_point_json(my_url, server_ip):
    # url = 'http://' + ip + ':' + port + '/api/v1/app/interface/cameraConfigs'
    url = my_url
    para = {"serverIp": server_ip}
    headers = {}

    r = requests.get(url, params=para, headers=headers)

    myjson = r.text
    myjson = json.loads(myjson)
    myjson = myjson['data']
    status = r.status_code
    print('get status code:', status)

    return myjson




