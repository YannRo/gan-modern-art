import os
import numpy as np
from PIL import Image
from tools.config import Config

config = Config()

def resize():
    # Defining image dir path. Change this if you have different directory
    images_path = config.IMAGE_DIR +'images/'

    training_data = []

    print('resizing...')

    for filename in os.listdir(images_path):
        path = os.path.join(images_path, filename)
        image = Image.open(path).resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.ANTIALIAS)

        training_data.append(np.asarray(image))

    training_data = np.reshape(
        training_data, (-1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS))
    training_data = training_data / 127.5 - 1

    name_file = 'cubism_data.npy'
    print(f'saving file at {os.getcwd()+name_file}')
    np.save(name_file, training_data)

if __name__=='__main__':
    resize()