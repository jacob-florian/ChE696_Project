import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heatmap

#Read data into dataframe
DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF = DF.drop(columns=['Element1', 'Element2'])
DF = DF.sort_values(by='Class')

#Read in SISSO Features
DF['sisso1'] = abs((DF['Allred-Rochow EN']+DF['Atomic radius ratio'])-(DF['Allred-Rochow EN']*DF['Martynov-Batsanov Mean EN']))
DF['sisso2'] = abs((DF['Allred-Rochow Mean EN']*DF['Zungerradius sum ratio']) - (DF['Martynov Ionic Character']/DF['Atomic radius ratio']))

#Define features and target
X = DF.iloc[:, list(range(2, 58))]
X_sisso = DF.iloc[:, list(range(58, 60))]
y = DF.iloc[:, 1]

################################# 30/70 Test/Train stratified split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20, stratify=y)

################################## Generate LDA Features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.fit_transform(X_test, y_test)

################################## Generate PCA Features
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

XLDA = [X_train_lda, X_test_lda, y_train, y_test]
XPCA = [X_train_pca, X_test_pca, y_train, y_test]

X_train, X_test, y_train, y_test = train_test_split(X_sisso, y, test_size=0.30, random_state=20, stratify=y)
XSISSO = [X_train, X_test, y_train, y_test]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
for X, feature_type in ((XSISSO, "SISSO"), (XPCA, "PCA"), (XLDA, "LDA")):
    parameters = {'min_samples_split':np.arange(2,3), 'max_depth':np.arange(2,10,2), 'max_leaf_nodes':np.arange(2,100,5)}
    tree = RandomForestClassifier(n_estimators=200)

    #Grid Search to find optimal parameters
    clf = GridSearchCV(tree, parameters, cv=5)
    clf.fit(X[0], X[2])
	
    #Print Cross Validation Scores
    #print("Best test score, " + feature_type + ":")
    #print(np.amax(clf.cv_results_.get("mean_test_score")))

    best_tree = clf.best_estimator_
    best_tree.fit(X[0], X[2])
    y_pred = best_tree.predict(X[1])
    y_pred_train = best_tree.predict(X[0])

    #Classification Report
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    print(feature_type + ' Train: ', accuracy_score(X[2], y_pred_train))
    print(feature_type + ' Test: ', accuracy_score(X[3], y_pred))



