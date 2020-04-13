#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: skozarekar
"""

from sklearn.decomposition import KernelPCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV


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
linear = KernelPCA(n_components=10, kernel="linear", random_state=42)
X_pca = StandardScaler().fit_transform(linear.fit_transform(X))

### LDA COMPONENTS ###
y = DF['Class'].to_numpy()
lda = LDA(n_components=5)
X_lda = StandardScaler().fit_transform(lda.fit(X, y).transform(X))

### TRAIN SVM CLASSIFIERS ON EACH X ###
from sklearn.svm import SVC

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
