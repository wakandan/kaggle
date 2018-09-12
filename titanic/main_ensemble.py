"""
logistic regression + random forest + svc = 0.73684
logistic regression + random forest + svc + decision tree = 0.75119
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
train.drop(['Cabin'], 1, inplace=True)
train = train.dropna()
y = train['Survived']
train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True)
train.fillna({'Age': 40, 'Fare': 35}, inplace=True)
X = pd.get_dummies(train)

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC()),
        ('tree', DecisionTreeClassifier()),
        ('knn', KNeighborsClassifier())
    ],
    voting='hard'
)

voting_clf.fit(X, y)


test = pd.read_csv('test.csv')
ids = test[['PassengerId']]
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
test.fillna({'Age': 40, 'Fare': 35}, inplace=True)
test = pd.get_dummies(test)
predictions = voting_clf.predict(test)
results = ids.assign(Survived=predictions)
results.to_csv('titanic_result_ensemble.csv', index=False)
