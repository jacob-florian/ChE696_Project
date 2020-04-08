#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:36:00 2020

@author: kruthis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import heatmap

#Read data into dataframe
DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]

DF = DF.drop(columns=['Element1', 'Element2'])
DF = DF.sort_values(by='Class')

#Define features and target
X = DF.iloc[:, list(range(2, 54))].values
y = DF.iloc[:, 1].values
y=y.astype(np.float64)

#30/70 Test/Train stratified split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20, stratify=y)


#Standardize Features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=6)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.fit_transform(X_test_std, y_test)

#Plot 2-D LDA
markers = ('s', 'x', 'o', '^', 'v','P','*')
#plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap=plt.cm.Paired, marker=markers[0])
#plt.xlabel('LDA 1')
#plt.ylabel('LDA 2')

#Declare empty arrays for accuracy and number of neigbours
accuracy= np.zeros((6,1))
accuracy_val = np.zeros((6,1))
neighbours= np.linspace(2,7,6)

for x in neighbours:
    
    
    #KNN model training
    from sklearn.neighbors import KNeighborsClassifier
    lr = KNeighborsClassifier(n_neighbors=int(x))
    
    
    #KNN Model
    lr = lr.fit(X_train_lda, y_train)
    #Classification Report
    from sklearn.metrics import classification_report,accuracy_score
    lr = lr.fit(X_train_lda, y_train)
    y_pred = lr.predict(X_train_lda)
    y_pred_test = lr.predict(X_test_lda)
    #print(classification_report(y_train, y_pred))
    accuracy[int(x-2),:] = accuracy_score(y_train, y_pred)
    accuracy_val[int(x-2),:] = accuracy_score(y_test, y_pred_test)
    print(accuracy_score(y_train, y_pred))

plt.plot(neighbours,accuracy,'-r',label='Training')
plt.plot(neighbours,accuracy_val,'-k',label='Validation')
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.title('Sensitivity study')
plt.legend()
