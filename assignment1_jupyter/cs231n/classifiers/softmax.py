from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D, C = W.shape
    N, _ = X.shape

    def soft(s):
        return np.exp(s)/np.sum(np.exp(s))
        

    for i in range(N):
        xW = X[i]@W
        
        prob = soft(xW)
        
        loss += -np.log(prob[y[i]])
        
        dW += X[i].reshape((-1,1)) @ prob.reshape((1,-1)) 
        dW[:,y[i]] -= X[i]
        # for k in range(D):
        #     for l in range(C):
        #         delta = 1 if y[i]==l else 0
        #         dW[k, l] += (prob[l]-delta)*X[i,k]
        # for j in range(dW.shape[1]):
        #   dW[:,j] += X[i] * prob[j]
        # dW[:,y[i]] -= X[i]
    
    loss /= N
    dW /= N
    
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W     
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, T=1):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    D, C = W.shape
    N, _ = X.shape
    
    XW = X@W
    exp_XW = np.exp(XW/T)
    
    denominador = np.sum(exp_XW, axis=1)
    numeradores = exp_XW
    numerador = numeradores[np.arange(len(y)), y]
    #import ipdb; ipdb.set_trace()
    salidas = -np.log(numerador/denominador)
    
    loss += salidas.sum()
    #print(salidas.shape)
    loss /= N
    loss += reg * np.sum(W * W)
    
    #import ipdb; ipdb.set_trace()
    soft_out = numeradores/denominador.reshape((-1,1))
    soft_out[np.arange(len(y)),y] -= 1
    dW += X.T @ soft_out
    
    dW /= N
    dW /= T
    dW += reg * 2 * W  
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW