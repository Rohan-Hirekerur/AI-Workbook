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

    def __init__(self, model_name: str):
        """
        Create instance of churn modeller
        Defines model arhitecture and locations to save model for :
        1. Label encoder
        2. One hot encoder
        3. Standard scalar
        4. Neural network - keras model

        Args:
            model_name (str): _description_
        """
        self.model_name = model_name

        # Derive and save paths to save / load data & models
        self.label_encoder_model_path = os.path.join(
            dir_name, 'models/label_encoder/{}.joblib'.format(model_name))
        self.one_hot_encoder_model_path = os.path.join(
            dir_name, 'models/one_hot_encoder/{}.joblib'.format(model_name))
        self.standard_scalar_model_path = os.path.join(
            dir_name, 'models/standard_scalar/{}.bin'.format(model_name))
        self.ann_model_path = os.path.join(
            dir_name, 'models/ann/{}'.format(model_name))

        # Initializing the ANN using keras sequential model
        self.ann = tf.keras.models.Sequential()

        # Adding the input layer and the first hidden layer
        # Note that first layer needs an iput shape to determine architecture
        # For layers and consequently, the model
        self.ann.add(tf.keras.layers.Dense(
            units=6, activation='relu', input_shape=(12,)))

        # Adding the second hidden layer
        self.ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

        # Adding the output layer
        self.ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # Compiling the ANN
        self.ann.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Print the model architecture
        self.ann.summary()

    def train(self, data_source: str, batch_size: int, epochs: int):
        """
        Train the churn modeller (ANN) and save the models
        for given data set batch size and number of epochs

        Args:
            data_source (str): Specify only the name & extension of dataset, it will be picked up from datasets directory, eg: bank_churn_data.csv
            batch_size (int): eg: 32
            epochs (int): eg: 100
        """

        # Load the dataset
        dataset: pd.DataFrame = pd.read_csv(os.path.join(
            dir_name, 'dataset/{}'.format(data_source)))

        # Name and surname of customer is irrelevant for determining churn.
        # Hence consider data from 4th column onwards (index 3)
        # Last column has data about churn, i.e. dependent variable. Add it to y
        X = dataset.iloc[:, 3:-1].values
        y = dataset.iloc[:, -1].values

        # Encoding categorical data
        # Label Encoding the "Gender" column
        # Gender is considered binary for this use case, hence using label encoder
        le: LabelEncoder = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])
        dump(le, self.label_encoder_model_path, compress=9)

        # One Hot Encoding the "Geography" column
        # As geography can have several values, using one-hot encoding instead of label endofing
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        dump(ct, self.one_hot_encoder_model_path)

        # Feature Scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)
        dump(sc, self.standard_scalar_model_path, compress=True)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        with tf.device('/CPU:0'):
            # Actual training begins here
            self.ann.fit(X_train, y_train,
                         batch_size=batch_size, epochs=epochs)
            self.ann.save(self.ann_model_path)

            # Predicting the Test set results
            y_pred = self.ann.predict(X_test)
            y_pred = (y_pred > self.threshold)

            # Making the Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix :")
            print(cm)
            accuracy_score(y_test, y_pred)

    def predict(self, X: list):
        """
        Predict if customers will churn as per input data

        Args:
            X (list): customer details

        Returns:
            [boolean]: Determining if customers will leave the bank
        """
        with tf.device('/CPU:0'):
            # Create a dataframe from the input list
            X = pd.DataFrame(X)
            X = X.iloc[:, 3:].values

            # Load saved ann model
            self.ann = tf.keras.models.load_model(self.ann_model_path)

            # Load label encoder and encode "Gender" column
            le: LabelEncoder = load(self.label_encoder_model_path)
            X[:, 2] = le.transform(X[:, 2])

            # Load One Hot Encoder and encode the "Geography" column
            ct = load(self.one_hot_encoder_model_path)
            X = np.array(ct.transform(X))

            # Load feature scaler and scale input
            sc = load(self.standard_scalar_model_path)
            X = sc.transform(X)

            probability = self.ann.predict(X)
            return probability > self.threshold
