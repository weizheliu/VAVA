"""VAVA Loss"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import numpy as np
import math


def vava_loss(X,Y,maxIter=20, lambda1=1.0,lambda2=0.1, virtual_distance = 5.0, zeta= 0.5, delta=2.0,global_step=None):
  """
  The vava loss term
    X: a N*d matrix, represent input N frames and each frame is encoded with d dimention vector
    Y: a M*d matrix, represent input M frames and each frame is encoded with d dimention vector
    maxIter: total number of iterations
    lambda1: the weight of the IDM regularization, default value: 1.0
    lambda2: the weight of the KL-divergence regularization, default value: 0.1
    virtual_distance: the threhold value to clip the distance for virtual frame,default value 5.0
    zeta: the theshold value for virtual frame, default value 0.5
    delta: the parameter of the prior Gaussian distribution, default value: 2.0
  """

  # We normalize the hyper-parameters based on the sequence length, and we found it brings better performance
  lambda1 = lambda1*(N+M)
  lambda2 = lambda2*(N*M)/4.0

  N = X.shape[0]
  M = Y.shape[0]

  D_x_y = tf.reduce_mean((tf.expand_dims(X, 1)-tf.expand_dims(Y, 0))**2,2)
  min_index = tf.math.argmin(D_x_y,axis=1)
  min_index = min_index.numpy()
  min_index = min_index.astype(np.float32)

  # add one more value for virtual frame, which is the first one
  N+=1
  M+=1

  # GMM 
  power = int(np.sqrt(global_step.numpy()+1.0))
  phi = 0.999**power
  phi = min(phi,0.999)
  phi = max(phi,0.001)

  P = np.zeros((N,M))
  mid_para = 1.0/(N**2)+1/(M**2)
  mid_para = np.sqrt(mid_para)
  pi = math.pi
  threshold_value = 2.0*virtual_distance/(N+M)
  for i in range(1,N+1):
    for j in range(1,M+1):
      # the distance to diagonal
      d_prior = np.abs(i/N-j/M)
      d_prior = d_prior/mid_para
      # the distance to the most similar matching for a giving i, adding extra 1 for virtual frame
      if i>1:
        d_similarity = np.abs(j/M-(min_index[i-2]+1)/M)
      else:
        d_similarity = np.abs(j/M-1.0/M)
      d_similarity = d_similarity/mid_para
      p_consistency = np.exp(-d_prior**2.0/(2.0*delta**2))/(delta*np.sqrt(2.0*pi))
      p_optimal = np.exp(-d_similarity**2.0/(2.0*delta**2))/(delta*np.sqrt(2.0*pi))
      P[i-1,j-1] = phi*p_consistency+(1.0-phi)*p_optimal
      # virtual frame prior value
      if (i == 1 or j == 1) and not(i==j):
        d = threshold_value*1.5/mid_para
        P[i-1,j-1] = np.exp(-d**2.0/(2.0*delta**2))/(delta*np.sqrt(2.0*pi))
  P = tf.convert_to_tensor(P,dtype=tf.float32)

  S = np.zeros((N,M))
  for i in range(1,N+1):
    for j in range(1,M+1):
      s_consistency = np.abs(i/N-j/M)
      if i>1:
        s_optimal = np.abs(j/M-(min_index[i-2]+1)/M)
      else:
        s_optimal = np.abs(j/M-1.0/M)
      s_consistency = lambda1/(s_consistency**2+1.0)
      s_optimal = lambda1/(s_optimal**2+1.0)
      S[i-1,j-1] = phi*s_consistency+(1.0-phi)*s_optimal
      if (i == 1 or j == 1) and not(i==j):
        s = threshold_value*1.5
        S[i-1,j-1] = lambda1/(s**2+1.0)

  S = tf.convert_to_tensor(S,dtype=tf.float32)
  XX = tf.math.reduce_sum(tf.math.multiply(X,X),axis=1,keepdims=True)
  Y_transpose = tf.transpose(Y)
  YY = tf.math.reduce_sum(tf.math.multiply(Y_transpose,Y_transpose),axis=0,keepdims=True)
  XX = tf.tile(XX,[1,M-1])
  YY = tf.tile(YY,[N-1,1])
  D = XX+YY-2.0*tf.matmul(X,Y_transpose)
  bin1 = tf.constant(value=zeta,shape=[1,M-1])
  bin2 = tf.constant(value=zeta,shape=[N,1])
  D = tf.concat([bin1, D], 0)
  D = tf.concat([bin2, D], 1)

  K = tf.math.multiply(P,tf.math.exp((S-D)/lambda2))
  K = tf.clip_by_value(K, clip_value_min=1e-15, clip_value_max=1.0e20)

  a = tf.math.divide(tf.ones([N,1]),N)
  b = tf.math.divide(tf.ones([M,1]),M)

  ainvK = tf.math.divide(K,a)
  compt=0
  u = tf.math.divide(tf.ones([N,1]),N)
  while compt<maxIter:
    Ktu = tf.matmul(K,u,transpose_a=True)
    aKtu = tf.matmul(ainvK,tf.math.divide(b,Ktu))
    u = tf.math.divide(1.0,aKtu)
    compt=compt+1

  new_Ktu = tf.matmul(K,u,transpose_a=True)
  v = tf.math.divide(b,new_Ktu)

  aKv = tf.matmul(ainvK,v)
  u = tf.math.divide(1.0,aKv)

  U = tf.math.multiply(K,D)
  dis = tf.math.reduce_sum(tf.math.multiply(u,tf.matmul(U,v)))
  dis = dis/(N*M*1.0)
  return dis,U

def all_loss(X,Y, lambda3=2.0, delta=15.0,global_step=None,temperature=0.5):
  """
    X: a N*d matrix, represent input N frames and each frame is encoded with d dimention vector
    Y: a M*d matrix, represent input M frames and each frame is encoded with d dimention vector
    lambda3: the margin value 
    delta: the margin for intra sequence postive and negative pairs
    temperature: temperature value for inter-sequence contrastive loss

  """

  N = X.shape[0]
  M = Y.shape[0]
  assert X.shape[1] ==Y.shape[1], 'The dimensions of instances in the input sequences must be the same!'

  # for C(x)
  W_x_p = np.zeros((N,N))
  for i in range(N):
    for j in range(N):
      W_x_p[i,j] = 1.0/((i-j)**2+1.0)
  W_x_p = tf.convert_to_tensor(W_x_p,dtype=tf.float32)

  y_x = np.zeros((N,N))
  for i in range(N):
    for j in range(N):
      if np.abs(i-j)>delta:
        y_x[i,j]=1.0
      else:
        y_x[i,j]=0.0
  y_x = tf.convert_to_tensor(y_x,dtype=tf.float32)

  # for C(y)
  W_y_p = np.zeros((M,M))
  for i in range(M):
    for j in range(M):
      W_y_p[i,j] = 1.0/((i-j)**2+1.0)
  W_y_p = tf.convert_to_tensor(W_y_p,dtype=tf.float32)

  y_y = np.zeros((M,M))
  for i in range(M):
    for j in range(M):
      if np.abs(i-j)>delta:
        y_y[i,j]=1.0
      else:
        y_y[i,j]=0.0
  y_y = tf.convert_to_tensor(y_y,dtype=tf.float32)

  # compute the distance
  D_x = tf.reduce_mean((tf.expand_dims(X, 1)-tf.expand_dims(X, 0))**2,2)
  D_y = tf.reduce_mean((tf.expand_dims(Y, 1)-tf.expand_dims(Y, 0))**2,2)

  C_x = tf.reduce_mean(y_x*tf.math.maximum(0.0,lambda3-D_x)+(1.0-y_x)*W_x_p*D_x)
  C_y = tf.reduce_mean(y_y*tf.math.maximum(0.0,lambda3-D_y)+(1.0-y_y)*W_y_p*D_y)

  vava_dis,U = vava_loss(X,Y,global_step=global_step)
  U = U[1:,1:]
  X_best = tf.math.argmax(U,axis=1)
  X_worst = tf.math.argmin(U,axis=1)
  Y_best = tf.math.argmax(U,axis=0)
  Y_worst = tf.math.argmin(U,axis=0)

  best_distance = tf.math.reduce_mean( (X-tf.gather(Y,X_best))**2 + (Y-tf.gather(X,Y_best))**2)/temperature
  worst_distance = tf.math.reduce_mean( (X-tf.gather(Y,X_worst))**2 + (Y-tf.gather(X,Y_worst))**2)/temperature
  loss_inter = tf.nn.softmax_cross_entropy_with_logits([0,1], [best_distance, worst_distance])

  overall =  0.5*(C_x+C_y)+vava_dis/(N*M)+0.0001*loss_inter
  return overall

