"""
mushroom dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

data = pd.read_csv('mushrooms.csv')

# to find any missing values
# print(data.isnull().sum())

# label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

X = data.iloc[:, 1:23].values
y = data.iloc[:, 0].values

# feature scaling
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
X = standard.fit_transform(X)

# dimensionality reduction - pca
from sklearn.decomposition import PCA
pca = PCA()
X = pca.fit_transform(X)
expv = pca.explained_variance_

"""
# plot of variance
with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(22), expv, alpha=0.5, align='center', label='individual explained variance')
    plt.ylabel('Explained variance')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
# visualising data after PCA
N = data.values
pca = PCA(n_components=2)
X = pca.fit_transform(N)
plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1])
plt.show()
"""

# modifying the dataset
mod_pca = PCA(n_components=17)
X = mod_pca.fit_transform(X)

# separating training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# simple logistic regression
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, y_train)
yprob_LR = classifier_LR.predict_proba(X_test)[:, 1]
ypred_LR = np.where(yprob_LR > 0.5, 1, 0)


from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
cm_LR = confusion_matrix(y_test, ypred_LR)
acc_LR = accuracy_score(y_test, ypred_LR)
print("Confusion Matrix:\n", cm_LR)
print("Accuracy Score: ", acc_LR)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, yprob_LR)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

# plotting ROC Curve
plt.figure(figsize=(10, 10))
plt.title("Receiver Operating Characteristics")
plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='-')
plt.axis('tight')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()