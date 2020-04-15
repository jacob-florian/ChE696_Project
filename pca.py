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
DF = DF.sort_values(by='Class')
DF.to_csv('np.txt', sep=' ', index=False, header=True)

x = DF.iloc[:, list(range(2, 58))]
y = DF.iloc[:, 1]
X = StandardScaler().fit_transform(x)

components = np.arange(2,20,1)
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
    

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(components, test_scores, label="Validation")
ax.plot(components, train_scores, label="Training")
ax.set(xlabel='Number of Components', ylabel='Accuracy Score',
       title='PCA Analysis')
ax.legend(loc='lower right',fontsize='x-large')
fig.savefig("pca_analysis.png")


pca = KernelPCA(n_components=2, kernel='linear')
X_reduced = pca.fit_transform(X)
   

plt.figure(figsize=(10, 10))
#plt.title("2-Component PCA, Linear Kernel", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel("$PC_1$", fontsize=24)
plt.ylabel("$PC_2$", fontsize=24, rotation=0)
plt.savefig("pca.png")

'''
plt.figure(figsize=(10, 10))
index = 0
pc1 = X_reduced[:, 0]
pc2 = X_reduced[:, 1]

plt.scatter(pc1, pc2, c=y.to_numpy())

for i in range(708):
    plt.scatter(pc1[i], pc2[i], s=10)
 '''   
    
'''
for i in [1, 2, 3, 4, 5, 6, 7]:
	index = DF.index[DF['Class'] == i].tolist()
	x = X_reduced[index, 0]
	y = X_reduced[index, 1]
	plt.scatter(x, y, s=10)


plt.legend(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
'''