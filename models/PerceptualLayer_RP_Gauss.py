import os
import pickle
import numpy as np
from sklearn.random_projection import GaussianRandomProjection


'''
    Parent Adaptive (only Perceptual) Layer class implementing a Gaussian Random Projection
'''

class PerceptualLayer_RP(object):

    def __init__(self, img_shape=(84,84,3), autoencoder='atari_v4', prototype_length=20, frozen_weights=False):
        self.model = GaussianRandomProjection(n_components = prototype_length)
        self.reconstructed_img = np.zeros(img_shape)
        self.count_img = 0
        self.stored_imgs = []
        self.reconstruct_error = 100
        self.dimensions = prototype_length
        print("embedding dimensions: ", self.dimensions)
        self.frozen = frozen_weights
        self.model_type = autoencoder
        #self.autoencoder.trainable == False

    def learn_protoype(self, img):
        #print('img shape', img.shape)
        sample = img.reshape((1,img.shape[1]*img.shape[2]*img.shape[3]))
        #print('sample shape', sample.shape)
        prototype = self.model.fit_transform(sample)[0]
        return prototype

    def get_prototype(self, img):
        #print('img shape', img.shape)
        sample = img.reshape((1,img.shape[1]*img.shape[2]*img.shape[3]))
        #sample = img.reshape((1,img.shape[0]*img.shape[1]*img.shape[2]))
        #print('sample shape', sample.shape)
        prototype = self.model.transform(sample)[0]
        #print('prototype shape', prototype.shape)
        return prototype

    def get_reconstruct_error(self):
        #print("update img shape", img.shape)
        #x = img.reshape(1,84,84,3)
        #reconstruct_error = self.autoencoder.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=0).history['loss'][-1]
        #self.reconstruct_error = self.autoencoder.fit(img, img, epochs=epochs, batch_size=batch_size, verbose=0).history['loss'][-1]
        #reconstruct_error = self.autoencoder.fit(img, img, epochs=epochs, batch_size=batch_size, verbose=1).history['loss']
        return self.reconstruct_error

    def get_reconstructed_img(self, prototype):
        rec_img = self.model.inverse_transform(prototype)
        self.reconstructed_img = rec_img.reshape((84, 84, 4))
        #print("reconstructed_img shape", self.reconstructed_img.shape())
        return self.reconstructed_img

    def save_model(self, save_dir, ID):
        filename = ID+'_grp'+str(self.dimensions)+'_atari_v4-test.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load_model(self):
        file_path = os.path.abspath('./data/autoencoders/trained/random_projection_p'+str(self.dimensions)+'.sav')

        if os.path.exists(file_path):
            self.model = pickle.load(open(file_path, 'rb'))
            print('FILE random_projection_p'+str(self.dimensions)+'.sav LOADED')
        else:
            print('FILE DOES NOT EXIST')
            print('Failed to find file on filepath: ', file_path)
