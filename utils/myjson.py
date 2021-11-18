import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
    return ret_dic