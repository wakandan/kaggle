"""
Accuracy = 0.65550
"""
import pandas as pd
from sklearn import tree
from sklearn.ensemble.bagging import BaggingClassifier

train = pd.read_csv("train.csv")

train.drop(['Cabin'], 1, inplace=True)

train = train.dropna()
y = train['Survived']
train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True)
train.fillna({'Age': 30})
X = pd.get_dummies(train)

bag_clf = BaggingClassifier(
    tree.DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=200,
    bootstrap=True,  # True => bagging, False => pasting
    n_jobs=-1  # use all cores
)

bag_clf.fit(X, y)

test = pd.read_csv('test.csv')
ids = test[['PassengerId']]
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
test.fillna(2, inplace=True)
test = pd.get_dummies(test)
predictions = bag_clf.predict(test)
results = ids.assign(Survived=predictions)
results.to_csv('titanic_result_bagging.csv', index=False)
