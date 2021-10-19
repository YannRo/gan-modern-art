from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

from tools.build import build_generator,build_discriminator
from tools.image import save_images
from tools.config import Config

config = Config()

training_data = np.load('cubism_data.npy')

training_data = np.load(os.path.join('dirname', 'filename.npy'))

optimizer = Adam(1.5e-4, 0.5)
discriminator = build_discriminator(config)
discriminator.compile(loss='binary_crossentropy',
optimizer = optimizer, metrics=['accuracy'])
generator = build_generator(config)
random_input = Input(shape=(config.NOISE_SIZE,))
generated_image = generator(random_input)
discriminator.trainable = False
validity = discriminator(generated_image)
combined = Model(random_input, validity)
combined.compile(loss='binary_crossentropy',
optimizer=optimizer, metrics=['accuracy'])
y_real = np.ones((config.BATCH_SIZE, 1))
y_fake = np.zeros((config.BATCH_SIZE, 1))
fixed_noise = np.random.normal(0, 1, (config.PREVIEW_ROWS * config.PREVIEW_COLS, config.NOISE_SIZE))

for epoch in range(config.EPOCHS):
    idx = np.random.randint(0, training_data.shape[0], config.BATCH_SIZE)
    x_real = training_data[idx]
 
    noise= np.random.normal(0, 1, (config.BATCH_SIZE, config.NOISE_SIZE))
    x_fake = generator.predict(noise)
 
discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
discriminator_metric_generated = discriminator.train_on_batch(
    x_fake, y_fake)
 
discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
generator_metric = combined.train_on_batch(noise, y_real)

cnt = 1
if epoch % config.SAVE_FREQ == 0:
   save_images(cnt, fixed_noise,config)
   cnt += 1
   print(f'{epoch} epoch, Discriminator accuracy: {100*  discriminator_metric[1]}, Generator accuracy: {100 * generator_metric[1]}')