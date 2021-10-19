import os
import numpy as np
from PIL import Image
from tools.config import Config

config = Config()

# Defining image dir path. Change this if you have different directory
images_path = config.IMAGE_DIR 

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(
    training_data, (-1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('saving file...')
np.save('cubism_data.npy', training_data)