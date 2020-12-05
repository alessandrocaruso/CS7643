import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  
  x1 = x.reshape((x.shape[0],np.prod(x.shape[1:])))
  out = x1@w + b
  
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  
  x, w, b = cache
  dx, dw, db = None, None, None
  x_shape = x.shape
  
  x = x.reshape((x.shape[0],np.prod(x.shape[1:])))
  #print(x.shape)

  dx = dout @ w.T #N x D
  dx = dx.reshape((x_shape))
  
  dw = x.T @ dout 
  
  db = np.sum(dout , axis=0)
  
  
  
   
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0,x)
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  
  
  
  x_1 = np.zeros_like(x)
  x_1[x >= 0] = 1
  
   
  dx =  dout * x_1
  
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #Variables initialization
  stride, pad = conv_param['stride'],conv_param['pad']
  N,C,H,W = x.shape
  F,C, HH, WW = w.shape
  
  #padding inputs with relative pad
  x_pad = np.zeros((N,C,H+2,W+2))
  for i in range(N):
        right = np.pad(x[i],((0,0),(0,pad),(0,pad)), mode = 'constant', constant_values=0) #pad from the right &up 
        x_pad[i] = np.pad(right, ((0,0), (pad,0),(pad,0)), mode = 'constant', constant_values =0) #complete the padding
  x = x_pad
  
  assert (H + 2 * pad - HH) % stride == 0
  assert (W + 2 * pad - WW) % stride == 0
  H_out = 1 + (H + 2 * pad - HH) // stride
  W_out = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, H_out, W_out))
  
    

  for i in range(x.shape[0]):
    img = x[i]

    for f in np.arange(w.shape[0]):
        filtro = w[f]
     
   #     v1= 0 #rows
    #    v2= WW #4 #rows
     #   v3= 0 #columns
      #  v4 = HH #4 columns
        for h in np.arange(H_out):
            for j in range(W_out):
                v1 = h*stride
                v2 = h*stride + HH 
                v3 = j*stride
                v4 = j*stride + WW
                patch = img[:,v1:v2,v3:v4]
                #fil = filtro[:,:,:]
                out[i,f,h,j] = np.sum(patch*filtro) + b[f]
               
     
        

      
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  
  
  #usual initialization
    
  x,w,b,conv_param = cache
  N,C,H,W = x.shape
  F,C, HH, WW = w.shape
  dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
   
  

  stride, pad = conv_param['stride'],conv_param['pad']
    
  for i in range(N):
    dout_n = dout[i]
    dx_i = dx[i]
    for f in np.arange(w.shape[0]):
        filtro = w[f]
        for h in np.arange(dout_n.shape[1]):
            for j in np.arange(dout_n.shape[2]):
                v1 = h * stride
                v2 = h * stride + HH 
                v3 = j * stride
                v4 = j * stride + WW
                patch_out = dout_n[f,h,j] #3D
                dx_i[:,v1:v2,v3:v4] += filtro * patch_out
                dw[f,...] += patch_out * x[i,:,v1:v2,v3:v4]
                db[f] += patch_out*1 
                
                
    dx[i,...] = dx_i[:,:,:]
 
  dx = dx[:,:,1:-1,1:-1]
 


             
  
  
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  pool_height , pool_width , stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
  N ,C ,H ,W  = x.shape
  H_out = 1 + (H  - pool_height) // stride
  W_out = 1 + (W - pool_width) // stride
    
  out = np.zeros((N, C, H_out, W_out))
  
    

  for i in range(x.shape[0]):
    img = x[i]
    for h in np.arange(H_out):
        for j in range(W_out):
            v1 = h * stride
            v2 = h * stride + pool_height 
            v3 = j * stride
            v4 = j * stride + pool_width
            patch = img[:, v1:v2 , v3:v4]
            for c in range(C):
                patch_c = patch[c,:,:]
                out[i,c,h,j] = np.max(patch_c)

    
  
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x , pool_param = cache
  N ,C ,H ,W  = x.shape
  pool_height , pool_width , stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
  dx = np.zeros_like(x)
  
  for i in range(N):
    img = x[i]
    dout_n = dout[i]
    dx_i = dx[i]
    for h in np.arange(dout_n.shape[1]):
        for j in np.arange(dout_n.shape[2]):
            v1 = h * stride
            v2 = h * stride + pool_height
            v3 = j * stride
            v4 = j * stride + pool_width
            patch_img = img[:,v1:v2,v3:v4]
            dx_i_window = dx_i[:,v1:v2,v3:v4]
            for c in range(C):
                patch_img_2d = patch_img[c,:,:]
                m,n = np.where(patch_img_2d == np.max(patch_img_2d))[0] ,np.where(patch_img_2d == np.max(patch_img_2d))[1]
                dx_i_window[c,m,n] += dout_n[c,h,j]
                               
    dx[i,...] = dx_i[:,:,:]
  
  
  
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

