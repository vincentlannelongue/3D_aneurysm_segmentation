import os
import sys
import h5py
import numpy as np
from skimage import filters
import cv2



# 2 idées à implémenter :
# - thresholder les images pour ne garder que l'intérieur des vaisseaux sanguins
# - dézoomer certaines images (quand on crop) pour avoir plus de petits anévrismes dans la dataset

class DataGen():

  def __init__(self, init_im_dataset : np.array, init_lab_dataset : np.array):
    self._init_im_dataset = init_im_dataset
    self._init_lab_dataset = init_lab_dataset
    self._data_max_value = np.amax(init_im_dataset)
    self._shape_image = self._init_im_dataset.shape[1:]

    self.im_dataset = self._init_im_dataset
    self.lab_dataset = self._init_lab_dataset
   
  def _crop(self, dataset, crop_format):
    means = [self._shape_image[k]//2 for k in range(3)]
    cropped_dataset = dataset[:, 
                              means[0]-crop_format[0]//2 : means[0]+crop_format[0]//2,
                              means[1]-crop_format[1]//2 : means[1]+crop_format[1]//2, 
                              means[2]-crop_format[2]//2 : means[2]+crop_format[2]//2]
    return cropped_dataset


  def _normalize(self, dataset : np.array):
    '''
    Normalize the array values.
    '''
    return dataset/np.amax(dataset)

  def _rotate(self, dataset : np.array, k : int = 1):
    ''' 
    Apply a rotation of k*90 degrees around the thickness axis.
    Even we decide to crop and make cubic data, we won't use other axis of rotation, because of the anisotropy of raw images.
    '''
    rotated_dataset = np.rot90(dataset, k = k, axes = (2, 3))
    return rotated_dataset

  def _add_noise(self, dataset : np.array, noise_level : float = 0.03):
    '''
    Add Gaussian noise to an image.
    '''
    sigma = noise_level*self._data_max_value
    noise = np.random.normal(loc = 0, scale = sigma, size = dataset.shape )
    noisy_dataset = dataset + noise
    return noisy_dataset

  def _mirror(self, dataset : np.array, mirror_axis : int = 1):
    '''
    Flip images along the given axis
    '''
    mirrored_dataset = np.flip(dataset, axis = mirror_axis)
    return mirrored_dataset
  
  def _vessels_threshold(self, dataset : np.array):
    thresh_data = []
    for im in dataset:
      thresh_im = []
      threshold = filters.threshold_otsu(im)        
      ret, thresh_im = cv2.threshold(im, 0.8*threshold, 1, cv2.THRESH_BINARY)
      thresh_data.append(thresh_im)
    thresh_data = np.array(thresh_data)
    return thresh_data

  def _process_dataset_batch(self, 
                             im_batch : np.array,
                             lab_batch : np.array,
                             vessel : bool = False,
                             mirror : bool = False, 
                             rotate : bool = False, 
                             noise : bool = False,
                             normalize : bool = False,
                             ):
    
    '''
    Apply all of the desired transformations on a batch of images and their corresponding label.
    '''
    # vessel : factor 1
    if vessel :
      vessel_im_batch = self._vessels_threshold(dataset=im_batch)
      im_batch = np.concatenate((im_batch, vessel_im_batch), axis=0)

    # mirror : factor 8
    if mirror:
      for i in range(3):
        mirrored_im_batch = self._mirror(dataset = im_batch, mirror_axis = i + 1)
        mirrored_lab_batch = self._mirror(dataset = lab_batch, mirror_axis = i + 1)
        im_batch = np.concatenate((im_batch, mirrored_im_batch), axis=0)
        lab_batch = np.concatenate((lab_batch, mirrored_lab_batch), axis=0)

    # rotate : factor 4
    if rotate:
      rotated_im_batches = [self._rotate(dataset=im_batch, k=i) for i in range(4)]
      rotated_lab_batches = [self._rotate(dataset=lab_batch, k=i) for i in range(4)] 

      im_batch = np.concatenate(rotated_im_batches, axis=0)
      lab_batch = np.concatenate(rotated_lab_batches, axis=0)

    # last : add noise
    if noise:
      noisy_batches = [self._add_noise(dataset = im_batch, noise_level = k*0.01) for k in range(4)] 
      lab_batches = [lab_batch for k in range(4)] 

      im_batch = np.concatenate(noisy_batches, axis=0)
      lab_batch = np.concatenate(lab_batches, axis=0)
    
    # normalize
    if normalize:
      im_batch = self._normalize(dataset = im_batch)



    return im_batch, lab_batch

  def create_dataset(self, 
                     batch_size : int = 2,
                     crop : int = 0,
                     vessel : bool = False,
                     mirror : bool = False, 
                     rotate : bool = False, 
                     noise : bool = False,
                     normalize : bool = False, 
                     ):
    
    im_dataset = self._init_im_dataset
    lab_dataset = self._init_lab_dataset
    nb_im = im_dataset.shape[0]

    if crop:
      im_dataset = self._crop(dataset=im_dataset, crop_format=crop)
      lab_dataset = self._crop(dataset=lab_dataset, crop_format=crop)

    transformed_im_dataset = []
    transformed_lab_dataset = []
    nb_batch = nb_im//batch_size
    print("nb_batch", batch_size, nb_batch)

    for k in range(nb_batch):
      print(f"starting batch number {k}")
      im_batch = im_dataset[k*batch_size:(k+1)*batch_size] 
      lab_batch = lab_dataset[k*batch_size:(k+1)*batch_size] 
      augm_im_batch, augm_lab_batch = self._process_dataset_batch(
          im_batch=im_batch,
          lab_batch=lab_batch,
          vessel = vessel,
          mirror=mirror,
          rotate=rotate,
          noise=noise,
          normalize=normalize
      )
      transformed_im_dataset.append(augm_im_batch)
      transformed_lab_dataset.append(augm_lab_batch)

    im_batch = im_dataset[nb_batch*batch_size:] 
    lab_batch = lab_dataset[nb_batch*batch_size:] 
    # if im_batch:
    augm_im_batch, augm_lab_batch = self._process_dataset_batch(
        im_batch=im_batch,
        lab_batch=lab_batch,
        vessel=vessel,
        mirror=mirror,
        rotate=rotate,
        noise=noise, 
        normalize=normalize
    )
    transformed_im_dataset.append(augm_im_batch)
    transformed_lab_dataset.append(augm_lab_batch)

    augm_im_dataset = np.concatenate(transformed_im_dataset, axis=0)
    augm_lab_dataset = np.concatenate(transformed_lab_dataset, axis=0)
                                     
    # need to shuffle the data and label the same way
    return augm_im_dataset, augm_lab_dataset