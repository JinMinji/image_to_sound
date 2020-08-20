#load_ext autoreload
#autoreload 2

import os
from pycocotools.coco import COCO
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import clean_sentence, get_prediction, get_sentence

import urllib.request
import re

# GUI error solution
plt.get_backend()
plt.rcParams["backend"] = "TkAgg"
plt.switch_backend("TkAgg")

# Define a transform to pre-process the testing images
transform_test = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create the data loader
data_loader = get_loader(transform=transform_test,
                         mode='test')

# Obtain sample image before and after pre-processing
orig_image, image = next(iter(data_loader))
# Convert image from torch.FloatTensor to numpy ndarray
transformed_image = image.numpy()
# Remove the first dimension which is batch_size euqal to 1
transformed_image = np.squeeze(transformed_image)
transformed_image = transformed_image.transpose((1, 2, 0))

# Visualize sample image, before pre-processing
plt.figure(1)
plt.imshow(np.squeeze(orig_image))
plt.title('example image')
#plt.show()
# Visualize sample image, after pre-processing
plt.figure(2)
plt.imshow(transformed_image)
plt.title('transformed image')

#plt.show()

# Load the most recent checkpoint
checkpoint = torch.load(os.path.join('./models', 'model-10.pkl'))
#checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'))

# Specify values for embed_size and hidden_size - we use the same values as in training step
embed_size = 256
hidden_size = 512

# Get the vocabulary and its size
vocab = data_loader.dataset.vocab
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Move models to GPU if CUDA is available.
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

# Obtain the embedded image features.
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)
print('example output:', output)

assert (type(output) == list), "Output needs to be a Python list"
assert all([type(x) == int for x in output]), "Output should be a list of integers."
assert all([x in data_loader.dataset.vocab.idx2word for x in
            output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

sentence = clean_sentence(output, vocab)
print('example sentence:', sentence)

assert type(sentence) == str, 'Sentence needs to be a Python string!'

get_prediction(data_loader, encoder, decoder, vocab)

#get_prediction(data_loader, encoder, decoder, vocab)

#get_prediction(data_loader, encoder, decoder, vocab)

#get_prediction(data_loader, encoder, decoder, vocab)

#get_prediction(data_loader, encoder, decoder, vocab)

print("잘됨")

annotations = get_sentence(data_loader, encoder, decoder, vocab)
ann = "".join(annotations)

# 캡션 결과값 스트링으로 받아 넣기
print(ann)
#ann_list=ann[92:]
ann_list = ann.split('.')
print(ann_list)


#annotations = str(coco_caps.getAnns(anns))

# 캡션 결과값 스트링으로 받아 넣기
#print(annotations)

client_id = "9W_QKXEHmed9UGw7bP3h" # 개발자센터에서 발급받은 Client ID 값
client_secret = "fH13lDcmFG" # 개발자센터에서 발급받은 Client Secret 값
#client_id = "l5F5Tv2msEnhFlHVmFpL" # 개발자센터에서 발급받은 Client ID 값
#client_secret = "UbZHkpNFTA" # 개발자센터에서 발급받은 Client Secret 값

encText = urllib.parse.quote(str(ann_list))
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
    print(response_body.decode('utf-8'))
    print(re.findall("\'[가-힣\s.]+\'", response_body.decode('utf-8')))
    k = re.findall("\'[가-힣\s.]+\'", response_body.decode('utf-8'))
    #final_annotation = "".join(k)
    #print(final_annotation)

# 번역 후 결과값만 뽑아 출력하기

else:
    print("Error Code:" + rescode)

picks = k

result = ""

for pre_pick in picks:
    if pre_pick[-2] == '다':
        pick = pre_pick.strip("'")
        result += str(pick)+". "

    else:
        pick =pre_pick.strip("'")
        result += pick+'이다. '

print(result)


'''
###
result = ""
for pre_pick in picks:
        pick =pre_pick.strip("'")
        result += pick+'.  '

print(result)
###
'''



# --------------클로바 연동------------------

client_id = "fgb0hedmm0"
client_secret = "ZJZ1aiR1bBPofwP2uzbcJqj7NOfgxvdhBx6yeEYi"
encText = urllib.parse.quote(picks[0])
#encText = urllib.parse.quote(result)
data = "speaker=mijin&speed=0&text=" + encText;
url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"
request = urllib.request.Request(url)
request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
request.add_header("X-NCP-APIGW-API-KEY",client_secret)
response = urllib.request.urlopen(request, data=data.encode('utf-8'))
rescode = response.getcode()

image_num = 16
sound_label='u-%d.mp3'%image_num
#sound_label=path_name

if(rescode==200):
    print("TTS mp3 저장")
    response_body = response.read()

    with open(sound_label, 'wb') as f:
        f.write(response_body)
else:
    print("Error Code:" + rescode)



