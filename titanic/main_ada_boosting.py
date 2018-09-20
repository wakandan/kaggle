"""
ada boosting = 0.77511
ada boosting with sub featuers = 0.78468
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.grid_search import GridSearchCV   #Perforing grid search

train = pd.read_csv("train.csv")
y = train['Survived']
train = train.fillna({'Age': 30, 'Fare': 35})
train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True) # drop irrelevant columns
train['Embarked'] = train['Embarked'].astype('category')
train['Embarked'] = train['Embarked'].cat.codes
train['Sex'] = train['Sex'].astype('category')
train['Sex'] = train['Sex'].cat.codes

ada_boost_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=500,
    algorithm='SAMME', learning_rate=0.5
)
# cv = cross_val_score(ada_boost_clf, train.loc[:, ['Sex', 'Age', 'Pclass', 'Parch']], y, cv=5)
# print(f"CV Score: mean {np.mean(cv)}, std {np.std(cv)}, min {np.min(cv)}, max {np.max(cv)}")

# parameters = {'n_estimators': np.linspace(100, 500, 5, dtype=np.int), 'learning_rate': np.linspace(0.1, 0.5, 5)}
# gs = GridSearchCV(ada_boost_clf, param_grid=parameters, cv=5, n_jobs=5)
# gs.fit(train.loc[: , ['Sex', 'Age', 'Pclass', 'Parch']], y)
# print(gs.best_params_, gs.best_score_)
ada_boost_clf.fit(train.loc[: , ['Sex', 'Age', 'Pclass', 'Parch']], y)


test = pd.read_csv('test.csv')
ids = test[['PassengerId']]
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
test = test.fillna({'Age': 40, 'Fare': 35})
test['Embarked'] = test['Embarked'].astype('category')
test['Embarked'] = test['Embarked'].cat.codes
test['Sex'] = test['Sex'].astype('category')
test['Sex'] = test['Sex'].cat.codes
predictions = ada_boost_clf.predict(test.loc[: , ['Sex', 'Age', 'Pclass', 'Parch']])
results = ids.assign(Survived=predictions)
results.to_csv('titanic_result_ada_boosting_sub_features.csv', index=False)
