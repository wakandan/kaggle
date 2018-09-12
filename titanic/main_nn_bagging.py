"""
Accuracy = 0.65550
"""
import pandas as pd
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.externals import joblib

import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec


class Argmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return K.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def process_data(file_name):
    train = pd.read_csv(file_name)
    train.drop(['Cabin'], 1, inplace=True)
    y = None
    if 'Survived' in train:
        train = train.dropna()
        is_test = False
        y = train['Survived']
        train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True)
    else:
        is_test = True
        passenger_ids = train['PassengerId']
        train.drop(['PassengerId', 'Name', 'Ticket'], 1, inplace=True)
    train = train.fillna({'Age': 30, 'Fare': 35})
    train.to_csv('test_temp.csv')
    X = pd.get_dummies(train)
    fare_min_max_scaler = preprocessing.MinMaxScaler()
    X['Fare'] = fare_min_max_scaler.fit_transform(X['Fare'].reshape(-1, 1))
    age_min_max_scaler = preprocessing.MinMaxScaler()
    X['Age'] = age_min_max_scaler.fit_transform(X['Age'].reshape(-1, 1))
    if is_test:
        return X, passenger_ids
    else:
        return X, y


def build_model():
    model = Sequential([
        Dense(64, input_shape=(10,), activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])
    sgd = optimizers.SGD(lr=1e-5, decay=1e-8, momentum=0.9, nesterov=True)
    # For a binary classification problem
    model.compile(optimizer=sgd,
                  loss='mse',
                  metrics=['accuracy'])
    return model


def train_model(X, y):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    callback_list = [earlystop]
    model = build_model()
    model_wrapper = KerasClassifier(build_fn=build_model, epochs=1000, batch_size=32, validation_split=0.2,
                                    shuffle=True)
    # model.fit(X, y, epochs=10000, batch_size=32, validation_split=0.2, shuffle=True, callbacks=callback_list)
    bag_clf = BaggingClassifier(model_wrapper, n_estimators=200)
    # y = y.astype('float')
    bag_clf.fit(X, y)
    # joblib.dump(bag_clf, 'bagging_classifier.pkl')
    # model.save('mlp.h5')
    return bag_clf


def train():
    X, y = process_data("train.csv")
    return train_model(X, y)


def test(model):
    X_test, passenger_ids = process_data('test.csv')
    y_pred = model.predict(X_test)
    print(y_pred)
    res = pd.DataFrame()
    res['PassengerId'] = passenger_ids
    res['Survived'] = y_pred
    print(res)
    # results = results.assign(Survived=y_pred)
    res.to_csv('titanic_result_nn.csv', index=False)


if __name__ == '__main__':
    model = train()
    test(model)
    # process_data('./train.csv')
