import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split


path ="C:/Users/vince/dev/idsc/Anev/"
original_data_path = path + "/challenge_dataset"
cropped_data_path = path + "/cropped_dataset_32x96x96"

os.chdir(path)

filenames = [cropped_data_path + "/" + os.listdir(original_data_path)[k] for k in range(len(os.listdir(original_data_path)))]

from DataGen import DataGen
from models import Basic_3D_CNN, AutoEncoder, UNet

### ORIGINAL DATA

nb_files = len(filenames)

im_data = np.load(os.path.join(cropped_data_path +  '/cropped_augm_32x96x96_im_data.npy'))
lab_data = np.load(os.path.join(cropped_data_path +  '/cropped_augm_32x96x96_lab_data.npy'))


#display shape of the data
reshape_shape = (-1, 32, 96, 96, 1) 

print(im_data.shape)
print(lab_data.shape)
# print((nb_files))



### Build training data

in_shape = reshape_shape[1:]
print("in_shape: ", in_shape)

X_train, X_test, y_train, y_test = train_test_split(im_data, lab_data, test_size=0.8, random_state=42)

X_train = X_train.reshape(reshape_shape)
X_test = X_test.reshape(reshape_shape)
y_train = y_train.reshape(reshape_shape)
y_test = y_test.reshape(reshape_shape)

print(X_train.shape)

# cnn = Basic_3D_CNN
unet = UNet(shape=in_shape)
# autoencoder = AutoEncoder(shape=in_shape)
unet_model = unet.build()

unet_model.fit(x=X_train, y=y_train, epochs=5, batch_size=50)

print("fitted")

unet.save(path="models/")