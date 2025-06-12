import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam


# Convolutional Autoencoder as visual processing of the agent (generating compressed representations).
class Conv_Autoencoder():

    def __init__(self, shape=(84,84,3), pl=20):
        self.img_shape = shape
        self.encoder = 0
        self.decoder = 0
        self.pl
        self.optimizer = Adam(lr=0.01) #before 0.001
        self.autoencoder = self.build_model()
        self.autoencoder.compile(loss='mse', optimizer=self.optimizer)
        self.autoencoder.summary()

    def build_model(self):
        input_img = Input(shape=self.img_shape)
        filter_size = (3, 3)
        x = Conv2D(16, filter_size, activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(200)(x)
        encoded = Dense(self.pl, activation="sigmoid")(x)
        self.encoder = Model(input_img, encoded)

        decoder_input = Input(shape=(self.pl,))
        x = Dense(200)(decoder_input)
        x = Dense(11*11*32)(x)
        x = Reshape((11, 11, 32))(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, filter_size, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, filter_size, activation='sigmoid', padding='same')(x)
        self.decoder = Model(decoder_input, decoded)

        auto_input = Input(shape=self.img_shape)
        encoded = self.encoder(auto_input)
        decoded = self.decoder(encoded)
        autoencoder = Model(auto_input, decoded)

        return autoencoder

    def update(self, img, batch_size):
        reconstruct_error = self.autoencoder.fit(img, img, epochs=1, batch_size=batch_size).history['loss'][-1]
        return reconstruct_error

    def encode(self, img):
        prototype = self.encoder([img])[0]
        return prototype

    def decode(self, prototype):
        img = self.decoder([[prototype]])[0]
        return img

    def reconstruct_img(self, img):
        reconstructed_img = self.autoencoder([img])[0]
        return reconstructed_img

# Feedforward network acting as CRB, predicting next sensory state from actual (state,action) couplets.
class Feedforward(object):

    def __init__(self):
        self.input_shape = (25,)

        optimizer = Adam(lr=0.001)
        self.ff = self.build_model()
        self.ff.compile(loss='mse', optimizer=optimizer)
        self.ff.summary()

    def build_model(self):
        input = Input(shape=self.input_shape)
        x = Dense(100)(input)
        x = Dense(100)(x)
        x = Dense(20)(x)

        return Model(input, x)

    def predict(self, couplet, speed):
        return self.ff.predict([[np.concatenate([couplet, speed])]])[0]

    def update(self, previous_couplet, speed, prototype):
        previous_couplet = np.concatenate([np.concatenate(previous_couplet), speed.tolist()])
        prediction_error = self.ff.fit([[previous_couplet]], [[prototype]], epochs=1, batch_size=None).history['loss'][-1]
        return prediction_error


class AdaptiveLayer(object):

    def __init__(self, prototype_length=20):
        self.visual = Conv_Autoencoder(pl=prototype_length)
        self.crb = Feedforward()
        self.predicted_next_img = np.zeros((84,84,3))
        self.reconstructed_img = np.zeros((84,84,3))
        self.count_batch = 0
        self.stored_imgs = []
        self.reconstruct_error = 100
        self.prediction_error = 100
        self.batch_size = 16
        self.pl = prototype_length

    def advance(self, img, speed, action_RL):
        self.stored_imgs.append(img)
        self.count_batch += 1
        # Build Couplet as [Prototype, Action] vector.
        prototype = self.visual.encode(img)
        couplet = np.concatenate([prototype, action_RL])
        # Pass the Couplet through the CRB and store the predicted visual Image.
        predicted_next_prototype = self.crb.predict(couplet, speed)
        self.predicted_next_img = self.visual.decode(predicted_next_prototype)

        if self.count_batch == self.batch_size:
            self.reconstruct_error = self.visual.update(img, self.batch_size)
            self.count_batch = 0
            self.stored_imgs.clear()

        return self.predicted_next_img, self.reconstruct_error

    def update(self, previous_couplet, speed, prototype):
        self.prediction_error = self.crb.update(previous_couplet, speed, prototype)
        return self.prediction_error

    def get_prototype(self, img):
        prototype = self.visual.encode(img)
        return prototype

    def get_reconstructed_img(self, img):
        self.reconstructed_img = self.visual.reconstruct_img(img)
        return self.reconstructed_img

    def get_reconstruct_error(self):
        return self.reconstruct_error

    def get_predicted_next_img(self):
        return self.predicted_next_img      

    def get_prediction_error(self):
        return self.prediction_error

    def save_model(self, savePath, ID):
        self.visual.autoencoder.save(savePath+'/autoencoders/'+ID+'ae'+str(self.pl)+'.h5')

    def load_saved_model(self, gameID):

        file_path = os.path.abspath('./data/autoencoders/'+gameID+'/autoencoder_p'+str(self.pl)+'.h5')

        if os.path.exists(file_path):
            from keras.src.legacy.saving import legacy_h5_format

            AE = legacy_h5_format.load_model_from_hdf5(file_path, custom_objects={'mse': 'mse'})

            layer = AE.get_layer('dense_4')
            self.visual.encoder = Model(inputs=AE.inputs, outputs=layer.output)
            self.visual.decoder = Model(inputs=layer.output, outputs=AE.outputs)
            auto_input = Input(shape=self.visual.img_shape)
            encoded = self.visual.encoder(auto_input)
            decoded = self.visual.decoder(encoded)
            self.visual.autoencoder = Model(auto_input, decoded)
            self.visual.autoencoder.compile(loss='mse', optimizer=self.visual.optimizer)
            
            print('FILE '+gameID+' autoencoder_p'+str(self.pl)+'.h5 LOADED')
        else:
            print(f'FILE DOES NOT EXIST: {file_path}')

    '''def load_model(self):
        file_path = os.path.abspath('./data/autoencoders/trained/autoencoder_p'+str(self.pl)+'.h5')
        if os.path.exists(file_path):
            self.visual.autoencoder = load_model('./data/autoencoders/trained/autoencoder_p'+str(self.pl)+'.h5')
            print('FILE autoencoder_p'+str(self.pl)+'.h5 LOADED')
        else:
            print('FILE DOES NOT EXIST')'''
