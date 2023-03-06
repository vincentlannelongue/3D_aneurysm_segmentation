import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from keras.utils.vis_utils import plot_model
import cv2

from sklearn.model_selection import train_test_split

from models import UNet

### DATA LOADING

path = os.getcwd()
original_data_path = path + "/challenge_dataset"
cropped_data_path = path + "/cropped_dataset_20x28x28"

im_data = np.load(os.path.join(cropped_data_path + '/cropped_augm_20x28x28_im_data.npy'))
lab_data = np.load(os.path.join(cropped_data_path + '/cropped_augm_20x28x28_lab_data.npy')) 
reshape_shape = (-1, 20, 28, 28, 1) 
in_shape = reshape_shape[1:]

max = 14000

im_data = im_data[:max]
lab_data = lab_data[:max]

X, X_test, y, y_test = train_test_split(im_data, lab_data, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.11, random_state=42)

X_train = X_train.reshape(reshape_shape)
X_test = X_test.reshape(reshape_shape)
X_val = X_val.reshape(reshape_shape)
y_train = y_train.reshape(reshape_shape)
y_test = y_test.reshape(reshape_shape)
y_val = y_val.reshape(reshape_shape)


### MODEL TRAINING

Unet = UNet(filters = 8, shape=in_shape, lr=LR)
model = Unet.build()

plot_model(model, to_file=path + MODEL_NAME+".png" show_shapes=True, show_layer_names=True)

history = model.fit(x=X_train, 
                    y=y_train, 
                    validation_data=(X_val, y_val), 
                    batch_size=BATCHSIZE, 
                    epochs=EPOCHS)

### METRICS

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.savefig(f'plots/accuracy_{MODEL_NAME}.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Dice Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.savefig(f'plots/loss_{MODEL_NAME}.png')

plt.plot(history.history['IoU'])
plt.plot(history.history['val_IoU'])
plt.title('Mean IoU')
plt.ylabel('IoU')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.savefig(f'plots/IOU_{MODEL_NAME}.png')


### SAVE

model.save(path + '/' + f'model/unet_{MODEL_NAME}.h5')
print("model saved at " + path + '/' + f'model/unet_{MODEL_NAME}.h5')