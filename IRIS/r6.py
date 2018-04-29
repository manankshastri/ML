"""
IRIS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv('iris.csv')
X = d.iloc[:, 1:5].values
y = d.iloc[:, [5]].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# fitting svm
from sklearn.svm import SVC
classifier = SVC(C=1, kernel='poly', degree=1, coef0=0.0, random_state=0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X)
y_p1 = classifier.predict([[4.6, 3.6, 1.0, 0.2]])
y_p2 = classifier.predict([[6.1, 2.8, 4.0, 1.3]])
y_p3 = classifier.predict([[7.7, 3.0, 6.1, 2.3]])
y_p4 = classifier.predict([[3, 2, 4, 0.2]])
y_p5 = classifier.predict([[4.7, 3, 1.3, 0.2]])

print(y_p1)
print(y_p2)
print(y_p3)
print(y_p4)
print(y_p5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
print(cm)
print(acc)

# applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=10)
print(accuracies.mean())
print(accuracies.std())

#applying grid search - to find best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100], 'kernel': ['poly'], 'degree': [1, 2, 3], 'coef0':[0.0, 0.01]},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy)
print(best_parameters)



