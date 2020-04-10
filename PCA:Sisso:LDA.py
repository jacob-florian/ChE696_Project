#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:24:23 2020

@author: kruthis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read data into dataframe
DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]

DF = DF.drop(columns=['Element1', 'Element2'])
DF = DF.sort_values(by='Class')

X = DF.iloc[:, list(range(2, 54))].values
y = DF.iloc[:, 1].values
y=y.astype(np.float64)

#30/70 Test/Train stratified split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20, stratify=y) #CHANGE THE RANDOM STATE accordingly

#Standardize Features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#2 component PCA
from sklearn.decomposition import KernelPCA

linear = KernelPCA(n_components=10, kernel="linear", random_state=42)
rbf = KernelPCA(n_components=2, kernel="rbf", gamma=0.008)
sigmoid = KernelPCA(n_components=2, kernel="sigmoid", gamma=100)

X_reduced = linear.fit_transform(X)

#Sisso 
DF['sisso1'] = abs((DF['Allred-Rochow EN']+DF['Atomic radius ratio'])-(DF['Allred-Rochow EN']*DF['Martynov-Batsanov Mean EN']))
DF['sisso2'] = abs((DF['Allred-Rochow Mean EN']*DF['Zungerradius sum ratio']) - (DF['Martynov Ionic Character']/DF['Atomic radius ratio']))

DF['binary1'] = (DF['Allred-Rochow EN']*DF['Ionic radius ratio']) * (DF['Martynov-Batsanov Mean EN'] - DF['Ionic radius ratio'])
DF['binary2'] = (DF['Allred-Rochow EN']*DF['Group number sum']) * (DF['Atomic radius sum'] - DF['Crystal radius sum'])

#Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=6)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.fit_transform(X_test_std, y_test)