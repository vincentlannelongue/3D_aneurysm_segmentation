import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import cv2
from skimage import data, filters



def calc_norm(a):
  norm = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
  return norm

def distance(a, b, factors=[1, 1, 1]):
  vect = []
  for i in range (0, len(a)):
      vect.append(factors[i]*(a[i] - b[i]))
  return calc_norm(vect)


def create_sphere(r, shape):
  # pas isotrope (pixels de tailles différentes)
  max_dim = np.amax(shape)
  factors = shape/max_dim
  factors /= np.amax(factors)

  sphere = np.zeros(shape)
  center = [shape[i]//2 for i in range(len(shape))]
  for i in range(len(sphere)):
    for j in range(len(sphere[i])):
      for k in range(len(sphere[i, j])):
        if distance(center, [i, j, k]) <= r:
          sphere[i, j, k] = 1
  
  return sphere

def inter(im1, im2):
  intersection = im1*im2
  return np.sum(intersection)


def IOU(label, prediction):
  inter = label*prediction
  count_inter = np.sum(inter)
  count_label = np.sum(label)
  count_pred = np.sum(prediction)
  iou = count_inter/(count_label + count_pred - count_inter)
  return iou

def meanIOU(truth, pred):
  score = 0
  M = 0
  m = 1
  scores = []
  for i,x in enumerate(truth):
    iou = IOU(truth[i], pred[i])
    scores.append(iou)
    score+= iou
    if iou > M:
      M = iou
    if iou < m:
      m = iou
  return score/len(truth), M, m, scores


def get_threshold(im, lab):
  T = []
  for i in range(len(im)-1):
    for j in range(len(im[i])-1):
      for k in range(len(im[i][j])-1):
        # print("i, j, k : ", i, j, k)
        if lab[i, j, k]:
          T.append(im[i, j, k])
  threshold = filters.threshold_otsu(im)
  if min(T) < threshold:
    threshold = min(T)
  return threshold


def optimize_radius(thresh_im, shape):
  R = [i for i in range(10)]
  Spheres = [create_sphere(r=r, shape=shape) for r in R]
  IOUs = [IOU(thresh_im, sphere) for sphere in Spheres]
  m = 0
  idx = 0
  for i, iou in enumerate(IOUs):
    if iou > m:
      m = iou
      idx = i
  
  return R[idx]