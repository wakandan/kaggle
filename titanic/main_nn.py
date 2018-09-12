"""
Accuracy = 0.65550
"""
import pandas as pd
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from sklearn import preprocessing
import numpy as np


# x = df.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pandas.DataFrame(x_scaled)

def process_data(file_name):
    train = pd.read_csv(file_name)
    print(len(train))
    train.drop(['Cabin'], 1, inplace=True)
    y = None
    if 'Survived' in train:
        train = train.dropna()
        is_test = False
        y = train['Survived']
        labels = np.zeros((len(y), 2))
        for i, l in enumerate(y):
            labels[i, l] = 1
        y = labels
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


def train_model(X, y):
    model = Sequential([
        Dense(64, input_shape=(10,), activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax'),
    ])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    callback_list = [earlystop]
    sgd = optimizers.SGD(lr=1e-6, decay=1e-8, momentum=0.9, nesterov=True)
    # For a binary classification problem
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=10000, batch_size=32, validation_split=0.2, shuffle=True, callbacks=callback_list)
    model.save('mlp.h5')


def train():
    X, y = process_data("train.csv")
    train_model(X, y)


def test():
    X_test, passenger_ids = process_data('test.csv')
    model = load_model('mlp.h5')
    y_pred = model.predict(X_test)
    # print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    res = pd.DataFrame()
    res['PassengerId'] = passenger_ids
    res['Survived'] = y_pred
    print(res)
    # results = results.assign(Survived=y_pred)
    res.to_csv('titanic_result_nn.csv', index=False)


if __name__ == '__main__':
    # train()
    test()
    # process_data('./train.csv')
