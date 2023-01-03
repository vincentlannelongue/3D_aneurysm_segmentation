import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from DataGen import DataGen

path ="C:/Users/vince/dev/idsc/Anev/"
data_path = path + "/challenge_dataset"
os.chdir(path)

filenames = [data_path + "/" + os.listdir(data_path)[k] for k in range(len(os.listdir(data_path)))]

### ORIGINAL DATA

nb_files = len(filenames)

im_data = []
lab_data = []

for k in range(nb_files):
    filename = filenames[k]
    h5 = h5py.File(filename,'r')
    label = h5['label'] 
    raw = h5['raw'] 
    # imlabel = label[0,:,:]
    # imraw = raw[0,:,:]
    raw_im = raw[:, :, :]
    lab_im = np.array([label[k,:,:] for k in range(64)])

    #let us build a data set with all concatenated images
    
    im_data.append(raw_im)
    lab_data.append(lab_im)

im_data = np.array(im_data)  
lab_data = np.array(lab_data)


print(im_data.shape)
print(lab_data.shape)
print((nb_files))


### Creating augmented dataset with DataGen 

data_gen = DataGen(init_im_dataset=im_data, init_lab_dataset=lab_data)

im_cropped_data, lab_cropped_data = data_gen.create_dataset(
    batch_size = 10,
    crop = [32, 96, 96],
    mirror=True,
    rotate=True,
    # noise=True,
    normalize=True
    )

print("number of files in new db: ", len(im_cropped_data))

### Saving the dataset

save_path = path + "/cropped_dataset_32x96x96"

if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(os.path.join(save_path + '/cropped_augm_32x96x96_im_data.npy'), im_cropped_data)
np.save(os.path.join(save_path + '/cropped_augm_32x96x96_lab_data.npy'), lab_cropped_data)

print("augmented data saved")

