import model as m
import dataset as d
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Function for training
def training(epoch):
	# Set modes to train
	cnnEnc.train()
	rnnDec.train()

	for batch_idx, data in enumerate(train_loader):
		# Distribute data to device
		X, y=data
		X=X.to(device)
		y=y.to(device)

		optimizer.zero_grad()
		output=rnnDec(cnnEnc(X))

		loss=loss_criterion(output,y) # y is the target

		loss.backward()
		optimizer.step()

		print('Epoch:',(epoch+1),'Iteration:',(batch_idx+1),'Loss:', loss)



# Function for validation
def validation():

	# Set modes to eval
	cnnEnc.eval()
	rnnDec.eval()

	test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in valid_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnnDec(cnnEnc(X))

            loss = loss_criterion(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnnEnc.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnnDec.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))


# ---------------- PARAMETERS ----------------------------------
model_name='inception'
feature_extract=False
use_pretrained=True
cnn_encoding_length=2048
num_classes=2
num_epochs=100
learning_rate=1e-4
log_interval=10 # The interval at which the model will be saved

root_dir='../Data/'
#---------------------------------------------------------------

# Dataset loader
train_loader=Dataset_CRNN(root_dir=root_dir,)

# Define the cnn model
cnnEnc=m.initialize_model(model_name,cnn_encoding_length,feature_extract,use_pretrained) # To use pretrained model
# cnnEnc=MyModel() # To use your own model

# Define RNN decoder
rnnDec=m.DecoderRNN(CNN_embed_dim=cnn_encoding_length,h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=num_classes)

# Params to update
crnn_params=list(cnnEnc.parameters()) + list(rnnDec.parameters())

# Specify the loss to use
loss_criterion=F.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

# Specify the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the models to device
cnnEnc=cnnEnc.to(device)
rnnDec=rnnDec.to(device)

# Train the model
for epoch in range(num_epochs):
	training()
	validation()
