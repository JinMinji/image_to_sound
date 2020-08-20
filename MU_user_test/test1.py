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
imgIds = coco.getImgIds(imgIds = [37777])

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


client_id = "l5F5Tv2msEnhFlHVmFpL" # 개발자센터에서 발급받은 Client ID 값
client_secret = "UbZHkpNFTA" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote(annotations)
data = "source=en&target=ko&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()

if(rescode==200):
    response_body = response.read()
#    print(response_body.decode('utf-8'))
    result = response_body.decode('utf-8')
    a = result.split('"')
#    print(a)
    print(a[27])

else:
    print("Error Code:" + rescode)