#matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
import os
import sys
import urllib.request
import re

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir='C:/Users/User/Desktop/coco/PythonAPI'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco=COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [324158])
image_num=37777
imgIds = coco.getImgIds(imgIds = [image_num])

img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image

I= io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
# load and display instance annotations
plt.figure(1)
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)
# load and display keypoints annotations
plt.figure(2)
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
# initialize COCO api for caption annotations
plt.figure(3)
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps = COCO(annFile)
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off');
#plt.show()

annotations=str(coco_caps.getAnns(anns))


# --------------파파고 연동------------------
'''
#annotations=str(coco_caps.showAnns(anns))
annotations=str(coco_caps.getAnns(anns))
# 캡션 결과값 스트링으로 받아 넣기

client_id = "l5F5Tv2msEnhFlHVmFpL" # 개발자센터에서 발급받은 Client ID 값
client_secret = "UbZHkpNFTA" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote(annotations)
# 받은 캡션 파파고 번역하기

data = "source=en&target=ko&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()

if(rescode==200):
    response_body = response.read()
    #print(response_body.decode('utf-8'))
    print(re.findall("\'[가-힣\s.]+\'", response_body.decode('utf-8')))
    k= re.findall("\'[가-힣\s.]+\'", response_body.decode('utf-8'))
    for final in k:
        print(final)

# 번역 후 결과값만 뽑아 출력하기

else:
    print("Error Code:" + rescode)
'''


# --------------클로바 연동------------------

client_id = "fgb0hedmm0"
client_secret = "ZJZ1aiR1bBPofwP2uzbcJqj7NOfgxvdhBx6yeEYi"
encText = urllib.parse.quote(annotations)
data = "speaker=mijin&speed=0&text=" + encText;
url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"
request = urllib.request.Request(url)
request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
request.add_header("X-NCP-APIGW-API-KEY",client_secret)
response = urllib.request.urlopen(request, data=data.encode('utf-8'))
rescode = response.getcode()

sound_label='test_%d.mp3'%image_num

if(rescode==200):
    print("TTS mp3 저장")
    response_body = response.read()

    with open(sound_label, 'wb') as f:
        f.write(response_body)
else:
    print("Error Code:" + rescode)
