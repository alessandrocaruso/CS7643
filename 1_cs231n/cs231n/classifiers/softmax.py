import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
    
  #b = np.random.rand(W.shape[0]).reshape(W.shape[0],1)
  num_images = X.shape[1]

  #W = np.hstack((W,b))
  dW = np.zeros_like(W)

  #v = np.ones(X.shape[1]).reshape(1,X.shape[1])

  #X = np.vstack((X,v))
  

  scores_z = W.dot(X)

  unn_prob = np.exp(scores_z - np.max(scores_z))
  norm_prob = unn_prob/np.sum(unn_prob,axis = 0, keepdims = True)

  losses = -np.log(norm_prob[y,np.arange(num_images)])
    
      
  data_loss = np.sum(losses)/num_images
  reg_loss = reg*np.sum(W*W)
    
  loss = data_loss + reg_loss
  
  #compute gradient on scores
  grad_scores = norm_prob
  grad_scores[y,np.arange(num_images)] -= 1
  grad_scores /= num_images
  
  #dW
  dW = np.dot(grad_scores, X.T)

  #add regulariziation gradient
  db = 2*reg*W
  
  dW += db

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
