from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Reshape, Input, concatenate, BatchNormalization, Activation, Dropout, Dense, Flatten
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D


class Basic_3D_CNN:
    def __init__(self, in_shape):
        self._in_shape = in_shape

    def build(self):
        model = Sequential()

        model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=self._in_shape))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(BatchNormalization(center=True, scale=True))
        model.add(Dropout(0.5))

        model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=in_shape))
        model.add(UpSampling3D((2,2,2)))
        model.add(Conv3D(1, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=in_shape))



        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    
        model.summary()

class AutoEncoder:
    
    def __init__(self, shape, filters=16, latent_dim=2, lr=0.001, verbose=True):
        """
        Autoencoder with 3D convolutions.

        Arguments:
            shape: shape of the input image [size_x, size_y, size_z, 1]
            filters: number of filters of the first conv layer
            latent_dim = size of the latent space (dimensions of the reduced space)
            lr: learning rate
            verbose = Boolean, if True, will print information about the models
        """
        
        self.input_shape = shape
        self.latent_dim = latent_dim
        self.verbose = verbose

        # Build the Autoencoder Model
        layer_filters = [filters , filters*2] #, filters*4]

        # First build the Encoder Model
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = inputs
        # Stack of Conv2D blocks
        for filters in layer_filters:
            x = Conv3D(filters=filters,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        padding='same')(x)
            x = Conv3D(filters=filters,
                        kernel_size=3,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

        # Shape info needed to build Decoder Model
        last_shape = K.int_shape(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(self.latent_dim)(x)

        # Instantiate Encoder Model
        self.encoder = Model(inputs, latent)
        if self.verbose:
            self.encoder.summary()

        # Build the Decoder Model
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(last_shape[1] * last_shape[2] * last_shape[3] * last_shape[4])(latent_inputs)
        x = Reshape((last_shape[1], last_shape[2], last_shape[3], last_shape[4]))(x)
        ## NOTE : LES ACTIVATIONS ONT ETE CHANGEES A TANH DANS LES 
        ## DEUX CAS. AVANT IL N'Y AVAIT AUCUN DECODAGE (TOUT ETAIT
        ## REMIS A ZERO)
        # Stack of Transposed Conv2D blocks
        for filters in layer_filters[::-1]:
            x = Conv3DTranspose(filters=filters,
                                kernel_size=3,
                                strides=2,
                                activation='tanh',
                                padding='same')(x)
            x = Conv3D(filters=filters,
                                kernel_size=3,
                                strides=1,
                                activation='tanh',
                                padding='same')(x)                            

        # The activation of the output layer is a sigmoid, so that output values
        # are in the same range as input values
        outputs = Conv3D(filters=1,
                        kernel_size=3,
                        strides=1,
                        activation='tanh',
                        padding='same')(x)

        # Instantiate Decoder Model
        self.decoder = Model(latent_inputs, outputs)
        if self.verbose:
            self.decoder.summary()

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        self.autoencoder = Model(inputs, 
                                self.decoder(self.encoder(inputs)),
                                name='autoencoder')
        if self.verbose:
            self.autoencoder.summary()
        
        ## Compile it with an appropriate loss function
        loss_ = 'mse'
        self.autoencoder.compile(optimizer=Adam(learning_rate=lr), loss=loss_)
        self.autoencoder.summary()
        
    def fit(self, x, y, epochs, batch_size, validation_split=0.2):
        return self.autoencoder.fit(x=x, 
                             y=y, 
                             validation_split=validation_split,
                             shuffle=True, 
                             epochs=epochs, 
                             batch_size=batch_size)

    def predict(self, batch):
        """Autoencode batch of images"""
        return self.autoencoder.predict(batch)

    def save(self, path):
      self.autoencoder.save(path + '/my_autoencoder')
      self.encoder.save(path + '/my_encoder')
      self.decoder.save(path + '/my_decoder')

    def load_model(self, path):
      self.autoencoder = load_model(path + '/my_autoencoder')
      self.encoder = load_model(path + '/my_encoder')
      self.decoder = load_model(path + '/my_decoder')




class UNet:
    
    def __init__(self, shape, filters=8, lr=0.001, verbose=True):
        """
        U-Net with 3D convolutions.

        Arguments:
            shape: shape of the input image [size_x, size_y, size_z, 1]
            lr: learning rate
            verbose = Boolean, if True, will print information about the models
        """
        
        self.input_shape = shape
        self.filters = filters
        self.verbose = verbose
        self.lr = lr

    def _double_conv_block(self, x, n_filters):
        # Conv2D then ReLU activation
        x = Conv3D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        x = Conv3D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        
        return x

    def _downsample_block(self, x, n_filters):
        # returns p as well to concatenate later
        f = self._double_conv_block(x, n_filters)
        p = MaxPooling3D(2)(f)
        p = Dropout(0.3)(p)
        return f, p
    
    def _upsample_block(self, x, conv_features, n_filters):
      x = Conv3DTranspose(n_filters, 3, 2, padding="same")(x)
      x = concatenate([x, conv_features])
      x = Dropout(0.3)(x)
      # Conv2D twice with ReLU activation
      x = self._double_conv_block(x, n_filters)
      return x
    

    def build(self,):
        inputs = Input(shape=self.input_shape)
        # encoder: contracting path - downsample
        f1, p1 = self._downsample_block(inputs, self.filters)
        f2, p2 = self._downsample_block(p1, self.filters*2)
        #  bottleneck
        bottleneck = self._double_conv_block(p2, self.filters*4)
        # decoder: expanding path - upsample
        u4 = self._upsample_block(bottleneck, f2, self.filters*2)
        u5 = self._upsample_block(u4, f1, self.filters)

        # outputs
        outputs = Conv3D(1, 1, padding="same", activation = "softmax")(u5)

        # unet model with Keras Functional API
        unet_model = Model(inputs, outputs, name="U-Net")

        unet_model.compile(optimizer=Adam(learning_rate=self.lr), loss="mse")
        unet_model.summary()
        return unet_model
      

    def predict(self, batch):
        """Predict batch of images"""
        return self.model.predict(batch)

    def save(self, path):
      self.model.save(path + '/my_unet')


    def load_model(self, path):
      self.autoencoder = load_model(path + '/my_unet')