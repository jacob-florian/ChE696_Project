# Kernel PCA

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF = DF.drop(columns=['Element1', 'Element2', ])
DF = DF.sort_values(by='Class')
DF.to_csv('np.txt', sep=' ', index=False, header=True)

x = DF.iloc[:, list(range(2, 58))]
y = DF.iloc[:, 1]
X = StandardScaler().fit_transform(x)

### PCA ANALYSIS FOR DIFFERENT N-COMPONENTS ### 
components = np.arange(2,16,1)
test_scores = []
train_scores = []

for n in components:
    pca = KernelPCA(n_components=n, kernel="linear")
    X_reduced = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.20,random_state=42, stratify=y)
    
    clf = SVC(kernel='rbf', gamma='auto')
    clf.fit(X_train, y_train)
    
    train_preds = clf.predict(X_train)
    train_scores.append(accuracy_score(y_train, train_preds))
    
    test_preds = clf.predict(X_test)
    test_scores.append(accuracy_score(y_test, test_preds))
    
# plot accuracy vs n-components 
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(components, test_scores, label="Validation")
ax.plot(components, train_scores, label="Training")
ax.set(xlabel='Number of Components', ylabel='Accuracy Score',
       title='PCA Analysis')
ax.legend(loc='lower right',fontsize='x-large')
#fig.savefig("pca_analysis.pdf", dpi = 300)



### PLOTTING 2 PCs ###
pca = KernelPCA(n_components=2, kernel='linear')
X_reduced = pca.fit_transform(X)

DF['pca1'] = X_reduced[:, 0]
DF['pca2'] = X_reduced[:, 1]

plt.figure(figsize=(10, 10), dpi=400)
for i in [1, 2, 3, 4, 5, 6, 7]:
	index = DF.index[DF['Class'] == i].tolist()
	x = DF.loc[index, 'pca1']
	y = DF.loc[index, 'pca2']
	plt.scatter(x, y, s=40)

#plt.legend(['CsCl', 'NaCl', 'ZnS', 'CuAu', 'TlI', 'FeB', 'NiAs'], loc='lower right', fontsize='x-large')
plt.xlabel("$PC_1$", fontsize=24)
plt.ylabel("$PC_2$", fontsize=24, rotation=0)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
#plt.savefig("two_pcs.png")



### PLOTTING 2 LDs ###
y = DF['Class'].to_numpy()
lda = LDA(n_components=2)
X_lda = StandardScaler().fit_transform(lda.fit(X, y).transform(X))

DF['lda1'] = X_lda[:, 0]
DF['lda2'] = X_lda[:, 1]

plt.figure(figsize=(10, 10), dpi=400)
for i in [1, 2, 3, 4, 5, 6, 7]:
	index = DF.index[DF['Class'] == i].tolist()
	x = DF.loc[index, 'lda1']
	y = DF.loc[index, 'lda2']
	plt.scatter(x, y, s=40)

#plt.legend(['CsCl', 'NaCl', 'ZnS', 'CuAu', 'TlI', 'FeB', 'NiAs'], loc='lower right', fontsize='x-large')
plt.xlabel("$LD_1$", fontsize=24)
plt.ylabel("$LD_2$", fontsize=24, rotation=0)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
#plt.savefig("two_lds.png")
