import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from keras.utils.vis_utils import plot_model
import cv2

from sklearn.model_selection import train_test_split


path = os.getcwd()
original_data_path = path + "/challenge_dataset"
cropped_data_path = path + "/cropped_dataset_20x28x28"

os.chdir(path)

filenames = [cropped_data_path + "/" + os.listdir(original_data_path)[k] for k in range(len(os.listdir(original_data_path)))]

from DataGen import DataGen
from models import Basic_3D_CNN, AutoEncoder, UNet

### ORIGINAL DATA

nb_files = len(filenames)

im_data = np.load(os.path.join(cropped_data_path +  '/cropped_20x28x28_im_data.npy'))
lab_data = np.load(os.path.join(cropped_data_path +  '/cropped_20x28x28_lab_data.npy'))


#display shape of the data
reshape_shape = (-1, 20, 60, 60, 1) 

print(im_data.shape)
print(lab_data.shape)
# print((nb_files))


### Build training data

in_shape = reshape_shape[1:]
print("in_shape: ", in_shape)

mem_limit = 400

X_train, X_test, y_train, y_test = train_test_split(im_data[:mem_limit], lab_data[:mem_limit], test_size=0.2, random_state=42)

X_train = X_train.reshape(reshape_shape)
X_test = X_test.reshape(reshape_shape)
y_train = y_train.reshape(reshape_shape)
y_test = y_test.reshape(reshape_shape)

print(X_train.shape)





# cnn = Basic_3D_CNN
unet = UNet(shape=in_shape, filters=2)
# autoencoder = AutoEncoder(shape=in_shape)
unet_model = unet.build()

unet_model.fit(x=X_train, y=y_train, epochs=200, batch_size=50)

print("fitted")

unet.save(model=unet_model, path="models/")

pred = unet_model.predict(x = X_test[:10])
print(X_test[0].shape)

y_pred = pred[1].reshape(20, 60, 60)
y_test_ = y_test[1].reshape(20, 60, 60)
x_test_ = X_test[1].reshape(20, 60, 60)

# let us plot k images and their labels
k = 10
start = 5
fig, ax = plt.subplots(k,4, figsize=(100,100))
#let us have the image close to each other
fig.subplots_adjust(hspace=0.1, wspace=0.1)

y_pred = y_pred.reshape(y_pred.shape[0],60,60)
for i in range(k):
    ax[i,0].imshow(x_test_[start+i])
    ax[i,1].imshow(y_test_[start+i]) 
    ret, thresh_pred = cv2.threshold(y_pred[start+i], 0.5, 1,cv2.THRESH_BINARY )
    ax[i,3].imshow(thresh_pred)
    ax[i,2].imshow(y_pred[start+i])
plt.show()

