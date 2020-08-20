# Watch for any changes in vocabulary.py, data_loader.py, utils.py or model.py, and re-load it automatically.
#load_ext autoreload
#autoreload 2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import sys
from pycocotools.coco import COCO
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time

from utils import train, validate, save_epoch, early_stopping
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

# Set values for the training variables
batch_size = 32         # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 10          # number of training epochs

# Define a transform to pre-process the training images
transform_train = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Define a transform to pre-process the validation images
transform_val = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Build data loader, applying the transforms
train_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)
val_loader = get_loader(transform=transform_val,
                         mode='val',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)


# The size of the vocabulary
vocab_size = len(train_loader.dataset.vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

# Define the loss function
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=0.001)

# Set the total number of training and validation steps per epoch
total_train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
total_val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)
print ("Number of training steps:", total_train_step)
print ("Number of validation steps:", total_val_step)

#*********************Train for the first time version*********************
# Keep track of train and validation losses and validation Bleu-4 scores by epoch
train_losses = []
val_losses = []
val_bleus = []
# Keep track of the current best validation Bleu score
best_val_bleu = float("-INF")

start_time = time.time()
for epoch in range(4, num_epochs + 1):
    train_loss = train(train_loader, encoder, decoder, criterion, optimizer,
                       vocab_size, epoch, total_train_step)
    train_losses.append(train_loss)
    val_loss, val_bleu = validate(val_loader, encoder, decoder, criterion,
                                  train_loader.dataset.vocab, epoch, total_val_step)
    val_losses.append(val_loss)
    val_bleus.append(val_bleu)
    if val_bleu > best_val_bleu:
        print ("Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
               format(best_val_bleu, val_bleu))
        best_val_bleu = val_bleu
        filename = os.path.join("./models", "best-model.pkl")
        save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
                   val_bleu, val_bleus, epoch)
    else:
        print ("Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(epoch))
    # Save the entire model anyway, regardless of being the best model so far or not
    filename = os.path.join("./models", "model-{}.pkl".format(epoch))
    save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
               val_bleu, val_bleus, epoch)
    print ("Epoch [%d/%d] took %ds" % (epoch, num_epochs, time.time() - start_time))
    if epoch > 5:
        # Stop if the validation Bleu doesn't improve for 3 epochs
        if early_stopping(val_bleus, 3):
            break
    start_time = time.time()

'''


#*********************resume version*********************
# Load the last checkpoints
checkpoint = torch.load(os.path.join('./models', 'train-model-76500.pkl'))

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Load start_loss from checkpoint if in the middle of training process; otherwise, comment it out
start_loss = checkpoint['total_loss']
# Reset start_loss to 0.0 if starting a new epoch; otherwise comment it out
#start_loss = 0.0

# Load epoch. Add 1 if we start a new epoch
epoch = checkpoint['epoch']
# Load start_step from checkpoint if in the middle of training process; otherwise, comment it out
start_step = checkpoint['train_step'] + 1
# Reset start_step to 1 if starting a new epoch; otherwise comment it out
#start_step = 1

# Train 1 epoch at a time due to very long training time
train_loss = train(train_loader, encoder, decoder, criterion, optimizer,
                   vocab_size, epoch, total_train_step, start_step, start_loss)
                   


#*****************************************[8]*************************************************#

# Load checkpoints
train_checkpoint = torch.load(os.path.join('./models', 'train-model-712900.pkl'))
epoch_checkpoint = torch.load(os.path.join('./models', 'model-6.pkl'))
best_checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'))

# Load the pre-trained weights and epoch from the last train step
encoder.load_state_dict(train_checkpoint['encoder'])
decoder.load_state_dict(train_checkpoint['decoder'])
optimizer.load_state_dict(train_checkpoint['optimizer'])
epoch = train_checkpoint['epoch']

# Load from the previous epoch
train_losses = epoch_checkpoint['train_losses']
val_losses = epoch_checkpoint['val_losses']
val_bleus = epoch_checkpoint['val_bleus']

# Load from the best model
best_val_bleu = best_checkpoint['val_bleu']

train_losses.append(train_loss)
print (train_losses, val_losses, val_bleus, best_val_bleu)
print ("Training completed for epoch {}, saving model to train-model-{}.pkl".format(epoch, epoch))
filename = os.path.join("./models", "train-model-{}.pkl".format(epoch))
save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
           best_val_bleu, val_bleus, epoch)
                    
'''
#*****************************************[7]*************************************************#

'''
# Load the last checkpoint
checkpoint = torch.load(os.path.join('./models', 'val-model-75500.pkl'))

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Load these from checkpoint if in the middle of validation process; otherwise, comment them out
start_loss = checkpoint['total_loss']
start_bleu = checkpoint['total_bleu_4']
# Reset these to 0.0 if starting validation for an epoch; otherwise comment them out
#start_loss = 0.0
#start_bleu = 0.0

# Load epoch
epoch = checkpoint['epoch']
# Load start_step from checkpoint if in the middle of training process; otherwise, comment it out
start_step = checkpoint['val_step'] + 1
# Reset start_step to 1 if starting a new epoch; otherwise comment it out
#start_step = 1

# Validate 1 epoch at a time due to very long validation time
val_loss, val_bleu = validate(val_loader, encoder, decoder, criterion,
                              train_loader.dataset.vocab, epoch, total_val_step,
                              start_step, start_loss, start_bleu)
                              
                              
                              
#*****************************************[8]*************************************************#

# Load checkpoints
checkpoint = torch.load(os.path.join('./models', 'train-model-7.pkl'))
best_checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'))

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Load train and validation losses and validation Bleu-4 scores
train_losses = checkpoint['train_losses']
val_losses = checkpoint['val_losses']
val_bleus = checkpoint['val_bleus']
best_val_bleu = best_checkpoint['val_bleu']

# Load epoch
epoch = checkpoint['epoch']

val_losses.append(val_loss)
val_bleus.append(val_bleu)
print (train_losses, val_losses, val_bleus, best_val_bleu)

if val_bleu > best_val_bleu:
    print ("Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
           format(best_val_bleu, val_bleu))
    best_val_bleu = val_bleu
    print (best_val_bleu)
    filename = os.path.join("./models", "best-model.pkl")
    save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
               val_bleu, val_bleus, epoch)
else:
    print ("Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(epoch))
# Save the entire model anyway, regardless of being the best model so far or not
filename = os.path.join("./models", "model-{}.pkl".format(epoch))
save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
           val_bleu, val_bleus, epoch)
if epoch > 5:
    # Stop if the validation Bleu doesn't improve for 3 epochs
    if early_stopping(val_bleus, 3):
        print ("Val Bleu-4 doesn't improve anymore. Early stopping")

'''