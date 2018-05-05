"""
ionosphere dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = pd.read_csv(ionosphere.csv)
X = d.iloc[:, 1:35].values
y = d.iloc[:, 35].values

from sklearn.preprocessing import LabelEncoder
label_y = LabelEncoder()
y = label_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=151, random_state=0)

# classifier
from sklearn.svm import SVC
classifier = SVC(C=10, kernel='rbf', gamma=0.1, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc1 = accuracy_score(y_test, y_pred)
cm1 = confusion_matrix(y_test, y_pred)
print(acc1)
print(cm1)

# grid selection
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy)
print(best_parameters)