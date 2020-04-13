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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
# SVM SISSO
X_train, X_test, y_train, y_test = train_test_split(X_sisso, y, test_size=0.30, random_state=42)

from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=2, gamma='auto')
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("SVM Trained on SISSO:")
print(accuracy_score(y_test, preds))

# SVM PCA
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.30, random_state=42)

from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=2, gamma='auto')
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("SVM Trained on 10 PCs:")
print(accuracy_score(y_test, preds))

# SVM LDA
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.30, random_state=42)

from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=2, gamma='auto')
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("SVM Trained on 6 LDA:")
print(accuracy_score(y_test, preds))

 
