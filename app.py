# 1. Import libraries from Keras.
from signal import valid_signals
from keras.preprocessing import image
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential

# 2. Configure the CNN (Convolutional Neural Network).
classifier = Sequential()
# 2a. Convolution - extracting appropriate features from the input image.

# Non-Linearity (RELU) - replacing all negative pixel values in feature map by zero.
classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# 2b. Pooling: reduces dimensionality of the feature maps but keeps the most important information.
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2c. Adding a second convolutional layer and flattening in order to arrange 3D volumes into a 1D vector.
classifier.add(Flatten())

# 2d. Fully connected layers: ensures connections to all activations in the previous layer.
classifier.add(Dense(2064, activation='relu'))

classifier.add(Dense(864, activation='linear'))

# 3. Compile the CNN and train the classifier..
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')


classifier.fit(
    training_set,
    steps_per_epoch=20,
    epochs=3,
    validation_data=test_set,
    validation_steps=80
)

# 5. Generate predictions
test_image = image.load_img(
    'download.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction = ' This is a Cat'
else:
    prediction = 'This is a Dog'
# else:
#    prediction = 'Invoice'


image = Image.open('download.png')
plt.imshow(image)

print(prediction)
