import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        self.im_size = im_size
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_classes = n_classes
        self.pool = 2
        
        
        self.conv1 = nn.Conv2d(in_channels = self.im_size[0], out_channels= self.hidden_dim, kernel_size= self.kernel_size, padding = 1, stride = 1) #from NxC1xHxW to NxC2xH'xW'
        
        self.fc1 = nn.Linear(self.hidden_dim * ((self.im_size[1] + 2*self.pool - self.kernel_size)//self.pool)* ((self.im_size[2] + 2*self.pool - self.kernel_size)//self.pool) , self.n_classes)        
        
        
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        images = F.relu(self.conv1(images))
        images = F.max_pool2d(images, kernel_size = 2, stride = 2)
        
        images = images.view(images.shape[0],-1)
        
        scores = F.softmax(self.fc1(images))
        
        
        
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

