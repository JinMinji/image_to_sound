import os
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
dataDir = 'C:/Users/User/Desktop/image_captioning-master/'
dataType = 'val2014'
#instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
#captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids
ids = list(coco.anns.keys())

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# matplotlib inline

plt.get_backend()
plt.rcParams["backend"] = "TkAgg"
plt.switch_backend("TkAgg")

# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)



