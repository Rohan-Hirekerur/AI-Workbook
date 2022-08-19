from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from joblib import dump, load
import os

dir_name = os.path.dirname(__file__)


class model:
    threshold = 0.5

    def __init__(self, model_name):
        self.model_name = model_name

        # Derive and save paths to save / load data & models
        self.label_encoder_path = os.path.join(
            dir_name, 'models/label_encoder/{}.joblib'.format(model_name))
        self.one_hot_encoder_path = os.path.join(
            dir_name, 'models/one_hot_encoder/{}.joblib'.format(model_name))
        self.standard_scalar_path = os.path.join(
            dir_name, 'models/standard_scalar/{}.bin'.format(model_name))
        self.ann_path = os.path.join(
            dir_name, 'models/ann/{}'.format(model_name))

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

    def train(self, data_source: str, batch_size: int, epochs: int):
        dataset = pd.read_csv(os.path.join(
            dir_name, 'dataset/{}'.format(data_source)))
        X = dataset.iloc[:, 3:-1].values
        y = dataset.iloc[:, -1].values

        # Encoding categorical data
        # Label Encoding the "Gender" column
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])
        dump(le, self.label_encoder_path, compress=9)

        # One Hot Encoding the "Geography" column
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        dump(ct, self.one_hot_encoder_path)

        # Feature Scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)
        dump(sc, self.standard_scalar_path, compress=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        with tf.device('/CPU:0'):
            self.ann.fit(X_train, y_train,
                         batch_size=batch_size, epochs=epochs)

            # Print the architecture
            self.ann.summary()

            self.ann.save(self.ann_path)

            # Predicting the Test set results
            y_pred = self.ann.predict(X_test)
            y_pred = (y_pred > self.threshold)

            # Making the Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            accuracy_score(y_test, y_pred)

    def predict(self, X: list):
        # Load saved ann model
        self.ann = tf.keras.models.load_model(self.ann_path)

        # Load babel encoder and encode "Gender" column
        le = load(self.label_encoder_path)
        X[:, 2] = le.transform(X[:, 2])

        # Load One Hot Encoder and encode the "Geography" column
        ct = load(self.one_hot_encoder_path)
        X = np.array(ct.transform(X))

        # Load feature scaler and scale input
        sc = load(self.standard_scalar_path)
        X = sc.transform(X)

        probability = self.ann.predict(X)
        return probability > self.threshold
