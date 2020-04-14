#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shivani Kozarekar
"""

from sklearn.decomposition import KernelPCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF = DF.drop(columns=['Element1', 'Element2', ])
DF = DF.sort_values(by='Class')
DF.to_csv('np.txt', sep=' ', index=False, header=True)

### SISSO FEATURES ###
sisso1 = (abs((DF['Allred-Rochow EN']+DF['Atomic radius ratio'])-(DF['Allred-Rochow EN']*DF['Martynov-Batsanov Mean EN']))).to_numpy()
sisso2 = (abs((DF['Allred-Rochow Mean EN']*DF['Zungerradius sum ratio']) - (DF['Martynov Ionic Character']/DF['Atomic radius ratio']))).to_numpy()
X_sisso = StandardScaler().fit_transform(np.transpose(np.vstack((sisso1, sisso2))))

### PCA COMPONENTS ###
x = DF.iloc[:, list(range(2, 58))]
X = StandardScaler().fit_transform(x)
linear = KernelPCA(n_components=15, kernel="linear", random_state=42)
X_pca = StandardScaler().fit_transform(linear.fit_transform(X))

### LDA COMPONENTS ###
y = DF['Class'].to_numpy()
lda = LDA(n_components=6)
X_lda = StandardScaler().fit_transform(lda.fit(X, y).transform(X))

### TRAIN SVM CLASSIFIERS ON EACH X ###
from sklearn.svm import SVC
'''$
#grid search for optimal hyperparameters
for X, feature_type in ((X_sisso, "SISSO"), (X_pca, "PCA"), (X_lda, "LDA")):
    parameters = {'kernel':('linear', 'rbf'), 'C':np.arange(1,10,1), 'gamma':['auto', 'scale']}
    svc = SVC()
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X, y)
    temp = clf.cv_results_
    print("Best test score, " + feature_type + ":")
    print(np.amax(clf.cv_results_.get("mean_test_score")))
        

# SISSO params @ idx 25: {'C': 7, 'gamma': 'auto', 'kernel': 'rbf'}
# PCA params @ idx 0: {'C': 1, 'gamma': 'auto', 'kernel': 'linear'}
# LDA params @ idx 4: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
'''

lda = LDA(n_components=6)
X_lda = StandardScaler().fit_transform(lda.fit(X, y).transform(X))

X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.30, random_state=6, stratify=y)

svc = SVC(C=1, gamma='scale', kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)


c = classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'])
print(c)

