"""
GAN model architecture file.
Model can be modified here in create_model function.
"""
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate,Conv2DTranspose
from keras.layers import Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from yogan.library.utility.image_utils import combine_normalized_images, img_from_normalized_img
from keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, Activation,Dropout,\
    concatenate, Flatten, Lambda, Concatenate
from keras.optimizers import Adam
from keras import backend as K
import keras
import datetime
import numpy as np
from PIL import Image
import os
from keras.backend.tensorflow_backend import set_session
from yogan.library.utility.glove_loader import GloveModel
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


dt_string = datetime.datetime.now().strftime("%d%m%Y_%H%M_%S")

def generate_c(x):
    mean = x[:, :128]
    log_sigma = x[:, 128:]
    stddev = K.exp(log_sigma)
    epsilon = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c = stddev * epsilon + mean
    return c

class DCGan(object):
    '''
    DCGAN architecture class,
    the generator and disciminator architectures belong here.
    '''
    model_name = 'dc-gan'

    def __init__(self):
        K.common.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None
        self.img_width = 7#128
        self.img_height = 7#128
        self.img_channels = 3# 1 # 3
        self.random_input_dim = 100
        self.text_input_dim = 100
        self.config = None
        self.glove_source_dir_path = './glove_dir'
        self.glove_model = GloveModel()

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, DCGan.model_name + '-config.npy')

    @staticmethod
    def get_weight_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, DCGan.model_name + '-' + model_type + '-weights.h5')



    def create_model(self):
            '''
            Modify the generator and discriminator moodel architecture here.
            '''
            
            ## generator architecture
            init_img_width = self.img_width // 4
            init_img_height = self.img_height // 4
            random_input = Input(shape=(self.random_input_dim,))
            text_input1 = Input(shape=(self.text_input_dim,))
            random_dense = Dense(1024)(random_input)
            text_layer1 = Dense(1024)(text_input1)
            merged = concatenate([random_dense, text_layer1])
            generator_layer = LeakyReLU()(merged)

            generator_layer = Dense(256 * init_img_width * init_img_height)(generator_layer)
            generator_layer = BatchNormalization()(generator_layer)
            generator_layer = LeakyReLU()(generator_layer)
            generator_layer = Reshape((init_img_width, init_img_height, 256),
                                      )(generator_layer)
            

            # Deconvolution blocks for generator
            #block 1
            generator_layer = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(generator_layer)
            generator_layer = BatchNormalization()(generator_layer)
            generator_layer = LeakyReLU()(generator_layer)

            #block 2
            generator_layer = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(generator_layer)
            generator_layer = BatchNormalization()(generator_layer)
            generator_layer = LeakyReLU()(generator_layer)

            #block 3
            generator_layer = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False)(generator_layer)
            generator_layer = BatchNormalization()(generator_layer)
            generator_output = Activation('tanh')(generator_layer)


            self.generator = Model([random_input, text_input1], generator_output)

            learning_rate = 0.005

            g_optim = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
            self.generator.compile(loss='binary_crossentropy', optimizer=g_optim)



            with open('generator_arch_'+dt_string+'.txt','w') as f:
                self.generator.summary(print_fn=lambda x: f.write(x + '\n'))
            text_input2 = Input(shape=(self.text_input_dim,))
            text_layer2 = Dense(1024)(text_input2)
            img_input2 = Input(shape=(self.img_width, self.img_height, self.img_channels))

            ## discriminator architecture
            
            # Convolution blocks for discriminator
            #### block 1
            img_layer2 = Conv2D(64, kernel_size=5,strides=(2, 2), padding='same')(img_input2)
            img_layer2 = LeakyReLU()(img_layer2)
            img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)

            #### block 2
            img_layer2 = Conv2D(128, kernel_size=(5, 5),strides=(2, 2), padding='same')(img_layer2)
            img_layer2 = LeakyReLU()(img_layer2)
            img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)


            img_layer2 = Flatten()(img_layer2)
            img_layer2 = Dense(1024)(img_layer2)

            merged = concatenate([img_layer2, text_layer2])

            discriminator_layer = Activation('tanh')(merged)
            discriminator_layer = Dense(1)(discriminator_layer)
            discriminator_output = Activation('sigmoid')(discriminator_layer)

            self.discriminator = Model([img_input2, text_input2], discriminator_output)



            d_optim = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
            self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

            with open('discriminator_arch_'+dt_string+'.txt','w') as f:
                self.discriminator.summary(print_fn=lambda x: f.write(x + '\n'))
            model_output = self.discriminator([self.generator.output, text_input1])

            self.model = Model([random_input, text_input1], model_output)
            self.discriminator.trainable = False


            self.model.compile(loss='binary_crossentropy', optimizer=g_optim)


    def load_model(self, model_dir_path,model_name):
        '''
        Loads config, generator,discriminator, word embeddings here.
        '''
        config_file_path = DCGan.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.img_width = self.config['img_width']
        self.img_height = self.config['img_height']
        self.img_channels = self.config['img_channels']
        self.random_input_dim = self.config['random_input_dim']
        self.text_input_dim = self.config['text_input_dim']
        self.glove_source_dir_path = self.config['glove_source_dir_path']
        self.create_model()
        # Loads glove embeddings for text input.
        self.glove_model.load(self.glove_source_dir_path, embedding_dim=self.text_input_dim)
        ### loading all models
        ## naming is currently bad but will be changed because currently which date corresponds to which model is not known

        self.generator.load_weights(DCGan.get_weight_file_path(model_dir_path,'generator-'+model_name))
        self.discriminator.load_weights(DCGan.get_weight_file_path(model_dir_path,'discriminator-'+model_name))

    def fit(self, model_dir_path, image_label_pairs, epochs=None, batch_size=None, snapshot_dir_path=None,
            snapshot_interval=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        if snapshot_interval is None:
            snapshot_interval = 20

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path

        self.glove_model.load(data_dir_path=self.glove_source_dir_path, embedding_dim=self.text_input_dim)

        config_file_path = DCGan.get_config_file_path(model_dir_path)

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))
        text_batch = np.zeros((batch_size, self.text_input_dim))

        self.create_model()


        for epoch in range(epochs):
            batch_count = int(image_label_pairs.shape[0] / batch_size)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator

                image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                image_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    normalized_img = image_label_pair[0]
                    text = image_label_pair[1]
                    image_batch.append(normalized_img)
                    text_batch[index, :] = self.glove_model.encode_doc(text, self.text_input_dim)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)

                image_batch = np.array(image_batch)

                generated_images = self.generator.predict([noise, text_batch], verbose=0)

                if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
                    self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch, batch_index=batch_index)

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch([np.concatenate((image_batch, generated_images)),
                                                            np.concatenate((text_batch, text_batch))],
                                                           np.array([1] * batch_size + [0] * batch_size))
                print("Epoch %d batch %d d_loss : %f" % (epoch, batch_index, d_loss))

                # Step 2: train the generator
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))

                print("Epoch %d batch %d g_loss : %f" % (epoch, batch_index, g_loss))

            if epoch % 500==0:
                self.generator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'generator-epoch-'+str(epoch)+"-"+dt_string), True)
                self.discriminator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'discriminator-epoch-'+str(epoch)+"-"+dt_string), True)

        self.generator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'generator-'+dt_string), True)
        self.discriminator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'discriminator-'+dt_string), True)

    def generate_image_from_text(self, text):
        noise = np.zeros(shape=(1, self.random_input_dim))
        encoded_text = np.zeros(shape=(1, self.text_input_dim))
        encoded_text[0, :] = self.glove_model.encode_doc(text)
        noise[0, :] = np.random.uniform(-1, 1, self.random_input_dim)

        generated_images = self.generator.predict([noise, encoded_text], verbose=1)

        generated_image =generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))

    def save_snapshots(self, generated_images, snapshot_dir_path, epoch, batch_index):
        image = combine_normalized_images(generated_images)
        img_from_normalized_img(image).save(
            os.path.join(snapshot_dir_path, DCGan.model_name + '-' + str(epoch) + "-" + str(batch_index) +dt_string + ".png"))
