"""
ada boosting = 0.77511
"""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
train.drop(['Cabin'], 1, inplace=True)
train = train.dropna()
y = train['Survived']
train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True)
train.fillna({'Age': 40, 'Fare': 35}, inplace=True)
X = pd.get_dummies(train)

ada_boost_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm='SAMME', learning_rate=0.5
)

ada_boost_clf.fit(X, y)


test = pd.read_csv('test.csv')
ids = test[['PassengerId']]
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
test.fillna({'Age': 40, 'Fare': 35}, inplace=True)
test = pd.get_dummies(test)
predictions = ada_boost_clf.predict(test)
results = ids.assign(Survived=predictions)
results.to_csv('titanic_result_ada_boosting.csv', index=False)
