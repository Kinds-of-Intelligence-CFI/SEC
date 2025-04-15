import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop


# Convolutional Autoencoder as visual processing of the agent (generating compressed representations).
class Conv_Autoencoder():

    def __init__(self, shape=(84,84,3), pl=20):
        self.img_shape = shape
        self.encoder = 0
        self.decoder = 0
        self.pl = pl
        optimizer = RMSprop(learning_rate=0.001) #before 0.001
        self.autoencoder = self.build_model()
        self.autoencoder.compile(loss='mse', optimizer=optimizer)
        self.autoencoder.summary()

    def build_model(self):
        input_img = Input(shape=self.img_shape)
        filter_size = (3, 3)
        pooling_size = (2, 2)

        x = Conv2D(16, filter_size, activation='relu', padding='same')(input_img)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pooling_size, padding='same')(x)
        x = Flatten()(x)
        #x = Dense(200)(x)
        encoded = Dense(self.pl, activation="relu")(x)
        self.encoder = Model(input_img, encoded)

        decoder_input = Input(shape=(self.pl,))
        #x = Dense(200)(decoder_input)
        x = Dense(11*11*32)(decoder_input)
        x = Reshape((11, 11, 32))(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling_size)(x)
        x = Conv2D(16, filter_size, activation='relu')(x)
        x = UpSampling2D(pooling_size)(x)
        decoded = Conv2D(3, filter_size, activation='relu', padding='same')(x)
        self.decoder = Model(decoder_input, decoded)

        auto_input = Input(shape=self.img_shape)
        encoded = self.encoder(auto_input)
        decoded = self.decoder(encoded)
        autoencoder = Model(auto_input, decoded)

        return autoencoder

    def update(self, img, batch_size):
        #print(f"img shape: {img.shape}")
        img_batch = np.expand_dims(img, axis=0)
        reconstruct_error = self.autoencoder.fit(img_batch, img_batch, epochs=1, batch_size=batch_size, verbose=0).history['loss'][-1]
        return reconstruct_error

    def encode(self, img):
        #print(f"imge shape: {img.shape}")
        #print(f"expected shape: {self.img_shape}")
        #print(f"mdoel summary: {self.encoder.summary()}")
        img_batch = np.expand_dims(img, axis=0)
        prototype = self.encoder.predict([img_batch])[0]
        return prototype

    def decode(self, prototype):
        img = self.decoder.predict([[prototype]])[0]
        return img

    def reconstruct_img(self, img):
        reconstructed_img = self.autoencoder.predict([img])[0]
        return reconstructed_img

class PerceptualLayer(object):

    def __init__(self, prototype_length=20):
        self.visual = Conv_Autoencoder(pl=prototype_length)
        self.reconstructed_img = np.zeros((84,84,3))
        self.count_batch = 0
        self.stored_imgs = []
        self.reconstruct_error = 100
        self.batch_size = 16
        self.pl = prototype_length

    def advance(self, img):
        self.stored_imgs.append(img)
        self.count_batch += 1

        if self.count_batch == self.batch_size:
            self.reconstruct_error = self.visual.update(img, self.batch_size)
            self.count_batch = 0
            self.stored_imgs.clear()

    def get_prototype(self, img):
        prototype = self.visual.encode(img)
        return prototype

    def get_reconstructed_img(self, img):
        self.reconstructed_img = self.visual.reconstruct_img(img)
        return self.reconstructed_img

    def get_reconstruct_error(self):
        return self.reconstruct_error

    def save_model(self, savePath, ID):
        self.visual.autoencoder.save(savePath+'/autoencoders/'+ID+'ae'+str(self.pl)+'.h5')

    def load_saved_model(self, gameID):

        file_path = os.path.abspath('./data/autoencoders/'+gameID+'_autoencoder_p'+str(self.pl)+'.h5')

        if os.path.exists(file_path):
            self.visual.autoencoder = load_model(file_path)
            print('FILE '+gameID+' autoencoder_p'+str(self.pl)+'.h5 LOADED')
        else:
            print('FILE DOES NOT EXIST')

    '''def load_model(self):
        file_path = os.path.abspath('./data/autoencoders/trained/autoencoder_p'+str(self.pl)+'.h5')
        if os.path.exists(file_path):
            self.visual.autoencoder = load_model('./data/autoencoders/trained/autoencoder_p'+str(self.pl)+'.h5')
            print('FILE autoencoder_p'+str(self.pl)+'.h5 LOADED')
        else:
            print('FILE DOES NOT EXIST')'''
