from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    D, M = w.shape
    
    x_flat = x.reshape((N,-1))
    
    out = x_flat @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N = x.shape[0]
    
    dx = dout @ w.T
    dx = dx.reshape(x.shape)
    
    dw = x.reshape((N,-1)).T @ dout
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF OUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        # sample_mean = np.mean(x,axis=0,keepdims=True)
        # sample_var = np.var(x, axis=0, keepdims=True, ddof=1)
        #
        # x -= sample_mean
        # x /= (np.sqrt(sample_var)+eps)
        #
        # running_mean = momentum * running_mean + (1 - momentum)*sample_mean
        # running_var = momentum * running_var + (1 - momentum) * sample_var
        #
        # x *= gamma
        # x += beta
        #
        # out = x
        
        # FORWARD PASS: Step-by-Step
        
        # Step 1. m = 1 / N \sum x_i
        m = np.mean(x, axis=0, keepdims=True)
        
        # Step 2. xc = x - m
        xc = x - m
        
        # Step 3. xc2 = xc ^ 2
        xcsq = xc ** 2
        
        # Step 4. v = 1 / N \sum xc2_i
        v = np.mean(xcsq, axis=0, keepdims=True)
        
        # Step 5. vsq = sqrt(v + eps)
        vsqrt = np.sqrt(v + eps)
        
        # Step 6. invv = 1 / vsq
        invv = 1.0 / vsqrt
        
        # Step 7. xn = xc * invv
        xn = xc * invv
        
        # Step 8. xg = xn * gamma
        xgamma = xn * gamma
        
        # Step 9. out = xg + beta
        out = xgamma + beta
        
        cache = (x, xc, vsqrt, v, invv, xn, gamma, eps)
        
        running_mean = momentum * running_mean + (1 - momentum) * m
        running_var = momentum * running_var + (1 - momentum) * v


        # *****END OF OUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        xn = (x - running_mean) / np.sqrt(running_var + eps)
        
        out = gamma * xn + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF OUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (x, xc, vsqrt, v, invv, xn, gamma, eps) = cache
  
    N, D = x.shape
  
    # BACKWARD PASS: Step-byStep
  
    # Step 9. out = xg + beta
    dxg = dout
    dbeta = np.sum(dout, axis=0)
  
    # Step 8. xg = xn * gamma
    dxn = dxg * gamma
    dgamma = np.sum(dxg * xn, axis=0)
  
    # Step 7. xn = xc * invv
    dxc1 = dxn * invv
    dinvv = np.sum(dxn * xc, axis=0)
  
    # Step 6. invv = 1 / vsqrt
    dvsqrt = -1 / (vsqrt ** 2) * dinvv
  
    # Step 5. vsqrt = sqrt(v + eps)
    dv = 0.5 * dvsqrt / np.sqrt(v + eps)
  
    # Step 4. v = 1 / N \sum xcsq_i
    dxcsq = 1.0 / N * np.ones((N, D)) * dv
  
    # Step 3. xcsq = xc ^ 2
    dxc2 = 2.0 * dxcsq * xc
  
    # Step 2. xc = x - m
    dx1 = dxc1 + dxc2
    dm = - np.sum(dxc1 + dxc2, axis=0, keepdims=True)
  
    # Step 1. m = 1 / N \sum x_i
    dx2 = 1.0 / N * np.ones((N, D)) * dm
  
    dx = dx1 + dx2

    # *****END OF OUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta




def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.uniform(size=x.shape)<p) / p
        out = x * mask
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    pad = conv_param.get('pad', 0)
    stride = conv_param.get('stride', 1)
    
    H_ = int(1 + (H + 2 * pad - HH) / stride)
    W_ = int(1 + (W + 2 * pad - WW) / stride)
    
    x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
    x_pad[:, :, pad:-pad, pad:-pad] = x
    
    out = np.zeros((N, F, H_, W_))
    
    
    for n in range(N): #n: numero de muestra
        for f in range(F): #f: numero de filtro
            for i in range(H_): #i: fila
                for j in range(W_): #j: columna
                    # 
                    u = x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    h = w[f, :, :, :]
                    out[n, f, i, j] = np.sum(u*h) + b[f]
            
        
    
    cache = (x, w, b, conv_param)
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    _, _, H_, W_ = dout.shape
    
    x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
    x_pad[:, :, pad:-pad, pad:-pad] = x
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    # Se agrega un gradiente auxiliar respecto a xpad por simplicidad
    dx_pad = np.zeros_like(x_pad) 
    
    # Se calcula el gradiente repecto de x,w y b respecto a d[n,f,i,j]
    for n in range(N): #n: numero de muestra
        for f in range(F): #f: numero de filtro
            for i in range(H_): #i: fila
                for j in range(W_): #j: columna
                    dl_dy = dout[n, f, i, j]
                    db[f] += dl_dy
                    dw[f] += dl_dy * x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += (
                                                                        dl_dy * w[f])
                    
    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    pool_height, pool_width, stride = (pool_param['pool_height'], 
                                       pool_param['pool_width'], 
                                       pool_param['stride'])
    H_ = int(1 + (H - pool_height) / stride)
    W_ = int(1 + (W - pool_width) / stride)
    
    out = np.empty((N, C, H_, W_))
    
    for n in range(N): #n: numero de muestra
        for c in range(C): #c: numero de canal
            for i in range(H_): #i: fila
                for j in range(W_): #j: columna 
                    region = x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    out[n, c, i, j] = np.max(region)
                    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    
    N, C, H, W = x.shape
    pool_height, pool_width, stride = (pool_param['pool_height'], 
                                       pool_param['pool_width'], 
                                       pool_param['stride'])
    H_ = int(1 + (H - pool_height) / stride)
    W_ = int(1 + (W - pool_width) / stride)
    
    dx = np.zeros_like(x)
    
    for n in range(N): #n: numero de muestra
        for c in range(C): #c: numero de canal
            for i in range(H_): #i: fila
                for j in range(W_): #j: columna 
                    region = x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    idx_max = np.unravel_index(region.argmax(), region.shape) #da el indice con el shape adecuado
                    idx_max += np.array([i*stride, j*stride])
                    dx[n, c, idx_max[0], idx_max[1]] += dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    # Se reshepean los vectores para poder aplicar el batch norm anterior    
    N, C, H, W = x.shape
    x_ = np.transpose(x, (0,2,3,1)).reshape((-1, C))
    out_, cache = batchnorm_forward(x_, gamma, beta, bn_param)
    out = np.transpose(out_.reshape((N, H, W, C)), (0,3,1,2))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    dout_ = np.transpose(dout, (0,2,3,1)).reshape((-1, C))
    dx_, dgamma, dbeta = batchnorm_backward(dout_, cache)
    dx = np.transpose(dx_.reshape((N, H, W, C)), (0,3,1,2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def bloque1_forward(x, eps):
    '''
    Se refier a un parte del forward de batch normalization
    '''
    # Step 1. m = 1 / N \sum x_i
    m = np.mean(x, axis=0, keepdims=True)
        
    # Step 2. xc = x - m
    xc = x - m
    
    # Step 3. xc2 = xc ^ 2
    xcsq = xc ** 2
    
    # Step 4. v = 1 / N \sum xc2_i
    v = np.mean(xcsq, axis=0, keepdims=True)
    
    # Step 5. vsq = sqrt(v + eps)
    vsqrt = np.sqrt(v + eps)
    
    # Step 6. invv = 1 / vsq
    invv = 1.0 / vsqrt
    
    # Step 7. xn = xc * invv
    xn = xc * invv
    
    out = xn
    
    cache = (x, xc, vsqrt, v, invv, xn, eps)
    
    return out, cache

def bloque1_backward(dout, cache):
    N, D = dout.shape
    
    x, xc, vsqrt, v, invv, xn, eps = cache
    
    # Step 7. xn = xc * invv
    dxc1 = dout * invv
    dinvv = np.sum(dout * xc, axis=0)
  
    # Step 6. invv = 1 / vsqrt
    dvsqrt = -1 / (vsqrt ** 2) * dinvv
  
    # Step 5. vsqrt = sqrt(v + eps)
    dv = 0.5 * dvsqrt / np.sqrt(v + eps)
  
    # Step 4. v = 1 / N \sum xcsq_i
    dxcsq = 1.0 / N * np.ones((N, D)) * dv
  
    # Step 3. xcsq = xc ^ 2
    dxc2 = 2.0 * dxcsq * xc
  
    # Step 2. xc = x - m
    dx1 = dxc1 + dxc2
    dm = - np.sum(dxc1 + dxc2, axis=0, keepdims=True)
  
    # Step 1. m = 1 / N \sum x_i
    dx2 = 1.0 / N * np.ones((N, D)) * dm
  
    dx = dx1 + dx2
    
    return dx

def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO (OPTIONAL): Implement the forward pass for spatial group           #
    # normalization.                                                          # 
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to train-time batch normalization       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    # Se usan fuertemente las ideas de batchnorm_forward y batchnorm_backward
    # Ademas se implementan un subnodo llamado "bloque1" que basicamente hace 
    # una parte interna de batchnorm.
    # En 
    # https://iie.fing.edu.uy/~cmarino/DLVis/assignment2/apuntes_spatial_groupnorm.pdf
    # se puede observar esquematicamente el razonamiento (es desporlijo, pero
    # es lo que se uso para pensarlo)
    
    N, C, H, W = x.shape
    
    #step 1
    xr = x.reshape((-1, int(C/G*H*W))).T
    #inverse: x=xr.T.reshape((N, C, H, W))
    
    #step2
    xb1, cache_b1 = bloque1_forward(xr, eps)
    
    #step3
    xn = xb1.T.reshape((N, C, H, W))
    
    #step 4
    xg = xn * gamma.reshape((1,-1,1,1))
    
    #step 5
    out = xg + beta.reshape((1,-1,1,1))
    
    cache = (x, xr, xb1, cache_b1, xn, xg, gamma, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO (OPTIONAL): Implement the backward pass for spatial group          #
    # normalization.                                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    
    (x, xr, xb1, cache_b1, xn, xg, gamma, G) = cache
    
    #step5
    dbeta = dout.sum((0,2,3), keepdims=True)
    dxg = dout
    
    #step4
    dgamma = np.sum(dxg*xn, (0,2,3), keepdims=True)
    dxn = dxg * gamma.reshape((1,-1,1,1))
    
    #step3
    dxb1 = dxn.reshape((-1, int(C/G*H*W))).T
    
    #step2
    dxr = bloque1_backward(dxb1, cache_b1)
    
    #step1
    dx = dxr.T.reshape((N, C, H, W))
   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
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
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
