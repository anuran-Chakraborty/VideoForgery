import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ---------------------------- Dataloader ----------------------------------
class Dataset_CRNN(data.Dataset):

	# Constructor
	def  __init__(self,root_dir,videos,labels,frames,transform=None):
		
		self.root_dir=root_dir # The root directory where the videos are stored
		self.videos=videos	# Each video is a separate folder consisting of image sequence
		self.labels=labels
		self.frames=frames
		self.transform=transform

	# Len function
	def __len__(self):

		# Returns number of videos
		return len(self.videos) 

	# Function to read one video
	def read_images(self, path, selected_folder, transform):

		X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    # get item function
    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.root_dir, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y

# ---------------------------- Dataloader ----------------------------------