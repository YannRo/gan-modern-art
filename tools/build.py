from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

def build_discriminator(config):
    
    image_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    
    model = Sequential([
        Conv2D(32, kernel_size=3, strides=2,input_shape=image_shape, padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(64, kernel_size=3, strides=2, padding='same'),
        ZeroPadding2D(padding=((0, 1), (0, 1))),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(128, kernel_size=3, strides=2, padding='same'),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(256, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(512, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    input_image = Input(shape=image_shape)
    validity = model(input_image)
    
    return Model(input_image, validity)

def build_generator(config):
    
    model = Sequential([
        Dense(4 * 4 * 256, activation='relu', input_dim=config.NOISE_SIZE),
        Reshape((4, 4, 256)),
        UpSampling2D(),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(momentum=0.8),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(momentum=0.8),
        Activation('relu')
    ])
    
    for _ in range(config.GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding='same'))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation('relu'))
    model.add(Conv2D(config.IMAGE_CHANNELS, kernel_size=3, padding='same'))
    model.add(Activation('tanh'))
    
    input = Input(shape=(config.NOISE_SIZE))
    generated_image = model(input)
    model.summary()
    
    return Model(input, generated_image)