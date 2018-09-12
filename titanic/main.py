import pandas as pd
from sklearn import tree, ensemble

train = pd.read_csv("train.csv")
train.drop(['Cabin'], 1, inplace=True)
train = train.dropna()
y = train['Survived']
train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True)
# train.fillna({'Age': 40})
X = pd.get_dummies(train)

dtc = tree.DecisionTreeClassifier()
dtc.fit(X, y)

test = pd.read_csv('test.csv')
ids = test[['PassengerId']]
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
test.fillna(2, inplace=True)
test = pd.get_dummies(test)
predictions = dtc.predict(test)
results = ids.assign(Survived=predictions)
results.to_csv('titanic_result_decision_tree.csv', index=False)
