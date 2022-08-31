
class model:
    def __init__(self):

        # Initialising the CNN
        cnn = tf.keras.models.Sequential()

        # Step 1 - Convolution
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                       activation='relu', input_shape=[64, 64, 3]))

        # Step 2 - Pooling
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Adding a second convolutional layer
        cnn.add(tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Step 3 - Flattening
        cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Step 5 - Output Layer
        cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # Part 3 - Training the CNN

        # Compiling the CNN
        cnn.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=['accuracy'])

    def train(self):
        cnn.fit(x=training_set, validation_data=test_set, epochs=25)
    def predict(self):
