# Kernel PCA

from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF = DF.drop(columns=['Element1', 'Element2', ])

DF.to_csv('np.txt', sep=' ', index=False, header=True)

x = DF.iloc[:, list(range(2, 58))]
y = DF.iloc[:, 1]
X = StandardScaler().fit_transform(x)

components = np.arange(2,15,1)
test_scores = []
train_scores = []

for n in components:
    pca = KernelPCA(n_components=n, kernel="linear")
    X_reduced = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.20, random_state=42, stratify=y)
    
    clf = SVC(kernel='rbf', gamma='auto')
    clf.fit(X_train, y_train)
    
    train_preds = clf.predict(X_train)
    train_scores.append(accuracy_score(y_train, train_preds))
    
    test_preds = clf.predict(X_test)
    test_scores.append(accuracy_score(y_test, test_preds))
    

fig, ax = plt.subplots()
ax.plot(components, test_scores, '-k', label="Validation")
ax.plot(components, train_scores, '-r', label="Training")
ax.set(xlabel='Number of Components', ylabel='Accuracy Score',
       title='PCA Analysis')
ax.legend(loc='lower right',fontsize='medium')


    
    

    
    
    
    
    
                                                    