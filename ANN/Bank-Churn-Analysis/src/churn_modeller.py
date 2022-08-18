from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import tensorflow as tf


class model:
    threshold = 0.5
    def __init__(self):
        # Initializing the ANN
        self.ann = tf.keras.models.Sequential()

        # Adding the input layer and the first hidden layer
        self.ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

        # Adding the second hidden layer
        self.ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

        # Adding the output layer
        self.ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # Compiling the ANN
        self.ann.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, data_source: str, batch_size: int, epochs: int, model_name: str):
        dataset = pd.read_csv(data_source)
        X = dataset.iloc[:, 3:-1].values
        y = dataset.iloc[:, -1].values
        X = self.pre_process(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        with tf.device('/CPU:0'):
            self.ann.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

            # Print the architecture
            self.ann.summary()

            # Predicting the Test set results
            y_pred = self.ann.predict(X_test)
            y_pred = (y_pred > self.threshold)

            # Making the Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            accuracy_score(y_test, y_pred)

    def predict(self, inputs: list, model_name: str):
        probability = self.ann.predict(self.sc.transform(inputs))
        return probability > self.threshold

    def pre_process(self, data: list):
        # Encoding categorical data
        # Label Encoding the "Gender" column
        le = LabelEncoder()
        data[:, 2] = le.fit_transform(data[:, 2])

        # One Hot Encoding the "Geography" column
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        data = np.array(ct.fit_transform(data))

        # Feature Scaling
        self.sc = StandardScaler()
        processed_data = self.sc.fit_transform(data)

        return processed_data
