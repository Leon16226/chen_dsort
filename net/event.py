

class Event(object):
    def __init__(self, cameraIp, timestamp,
                 roadId, roadName, code, subCode, dateTime, status, no, distance, picture,
                 targetType, xAxis, yAxis, height, width, prob,
                 miniPicture, carNo,
                 remark
                 ):
        self.cameraIp = cameraIp
        self.timestamp = timestamp
        self.events = [
            {
                "roadId": roadId,
                "roadName": roadName,
                "code": code,
                "subCode": subCode,
                "dateTime": dateTime,
                "status": status,
                "no": no,
                "distance": distance,
                "picture": picture,
                "coordinate": [
                    {
                        "targetType": targetType,
                        "xAxis": xAxis,
                        "yAxis": yAxis,
                        "height": height,
                        "width": width,
                        "prob": prob
                    }
                ],
                "carNoAI": {
                    "miniPicture": miniPicture,
                    "carNo": carNo
                },
                "remark": remark
            }
        ]