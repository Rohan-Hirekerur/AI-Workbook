# Importing the libraries
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

dir_name = os.path.dirname(__file__)


class model:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.cnn_model_path = os.path.join(
            dir_name, 'models/cnn/{}'.format(model_name))

        # Initialising the CNN
        self.cnn = tf.keras.models.Sequential()

        # Step 1 - Convolution
        # Note that first layer needs an iput shape to determine architecture
        # For layers and consequently, the model
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                            activation='relu', input_shape=[64, 64, 3]))

        # Step 2 - Pooling
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Adding a second convolutional layer
        self.cnn.add(tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Step 3 - Flattening
        self.cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Step 5 - Output Layer
        self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # Compiling the CNN
        self.cnn.compile(optimizer='adam', loss='binary_crossentropy',
                         metrics=['accuracy'])

        # Print the model architecture
        self.cnn.summary()

    def train(self, epochs: int):
        # Preprocessing the Training set
        train_datagen = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        training_set = train_datagen.flow_from_directory(
            'src/dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

        # Preprocessing the Test set
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(
            'src/dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

        self.cnn.fit(x=training_set, validation_data=test_set, epochs=epochs)
        self.cnn.save(self.cnn_model_path)

    def predict(self):
        test_image = image.load_img(
            'src/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        self.cnn = tf.keras.models.load_model(self.cnn_model_path)
        result = self.cnn.predict(test_image)
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        return prediction
