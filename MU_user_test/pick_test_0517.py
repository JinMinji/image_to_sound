import os
import sys
import urllib.request


picks = ["'주방 근처의 식탁에는 과일 그릇이 놓여 있다'", "'작은 부엌에는 다양한 가전제품과 테이블이 있다'", "'흰색으로 장식된 부엌과 식당'", "'테이블 위에 과일 그릇이 놓여 있는 부엌'"]

result = ""


for pre_pick in picks:
    if pre_pick[-2] == '.':
        pick = pre_pick.strip("'")
        result += str(pick)+" "

if result == "":
    for pre_pick in picks:
        pick =pre_pick.strip("'")
        result += pick+'.   '

print(result)


client_id = "fgb0hedmm0"
client_secret = "ZJZ1aiR1bBPofwP2uzbcJqj7NOfgxvdhBx6yeEYi"
encText = urllib.parse.quote(result)
data = "speaker=mijin&speed=0&text=" + encText;
url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"
request = urllib.request.Request(url)
request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
request.add_header("X-NCP-APIGW-API-KEY",client_secret)
response = urllib.request.urlopen(request, data=data.encode('utf-8'))
rescode = response.getcode()

sound_label = 'test_1.mp3'

if(rescode==200):
    print("TTS mp3 저장")
    response_body = response.read()

    with open(sound_label, 'wb') as f:
        f.write(response_body)
else:
    print("Error Code:" + rescode)


