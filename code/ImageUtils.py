import numpy as np
from matplotlib import pyplot as plt

import os
import pickle
import torch
from torchvision import transforms as tf

"""This script implements the functions for data augmentation
and preprocessing.
"""

def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE

    # Ensure the image has the correct shape for visualization
    image = np.transpose(image, [1, 2, 0])
    
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

## Here I have created a class named MyDataset which performs the tasks of function preprocess_image and parse_record.

class MyDataset(torch.utils.data.Dataset):
  # Characterizes a dataset for PyTorch
  def __init__(self, data, labels, training=False):
        'Initialization'
        self.labels = labels
        self.data = torch.tensor(data.reshape(data.shape[0],3,32,32)/255, dtype=torch.float32)
        self.training = training
        if training:
            self.transform = tf.Compose([
                    tf.RandomHorizontalFlip(),
                    # Here I have used these hard coded values because I found them on internet and these are correct values calculated for normalization of cifar10 dataset images as the training data is same.
                    tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                    tf.RandomCrop((32,32), 4, padding_mode='edge'),
                    tf.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.05, 20), value=0, inplace=False),
                    ])
        else:
            self.transform = tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

  def __len__(self):
        # Total number of samples
        return self.data.shape[0]

  def __getitem__(self, index):
        # Generates one sample of data
        x = self.data[index]
        X = self.transform(x)
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        else:
            return X

### END CODE HERE