"""
logistic regression + random forest + svc = 0.73684
logistic regression + random forest + svc + decision tree = 0.75119
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score

train = pd.read_csv("train.csv")
y = train['Survived']
train = train.fillna({'Age': 30, 'Fare': 35})
train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True) # drop irrelevant columns
train['Embarked'] = train['Embarked'].astype('category')
train['Embarked'] = train['Embarked'].cat.codes
train['Sex'] = train['Sex'].astype('category')
train['Sex'] = train['Sex'].cat.codes
train = train.loc[:, ['Sex', 'Age', 'Pclass', 'Embarked']]
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC()),
        ('tree', DecisionTreeClassifier(max_depth=1)),
        ('knn', KNeighborsClassifier())
    ],
    voting='hard',
)

cv = cross_val_score(voting_clf, train, y, cv=5)
print(f"CV Score: mean {np.mean(cv)}, std {np.std(cv)}, min {np.min(cv)}, max {np.max(cv)}")
# voting_clf.fit(train, y)
#
#
# test = pd.read_csv('test.csv')
# ids = test[['PassengerId']]
# test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
# test = test.fillna({'Age': 40, 'Fare': 35})
# test['Embarked'] = test['Embarked'].astype('category')
# test['Embarked'] = test['Embarked'].cat.codes
# test['Sex'] = test['Sex'].astype('category')
# test['Sex'] = test['Sex'].cat.codes
# test = test.loc[:, ['Sex', 'Age', 'Pclass', 'Parch']]
# predictions = voting_clf.predict(test)
# results = ids.assign(Survived=predictions)
# results.to_csv('titanic_result_voting.csv', index=False)
