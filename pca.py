# 2-component PCA

from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF = DF.drop(columns=['Element1', 'Element2', ])

DF.to_csv('np.txt', sep=' ', index=False, header=True)

x = DF.iloc[:, list(range(2, 58))]
y = DF.iloc[:, 1]
X = StandardScaler().fit_transform(x)

linear = KernelPCA(n_components=2, kernel="linear")
rbf = KernelPCA(n_components=2, kernel="rbf", gamma=0.008)
sigmoid = KernelPCA(n_components=2, kernel="sigmoid", gamma=100)

plt.figure(figsize=(11, 3))
for subplot, pca, title in ((133, sigmoid, "Sigmoid kernel"), (132, rbf, "RBF kernel"), (131, linear, "Linear kernel")):
    X_reduced = pca.fit_transform(X)
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hsv)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    #plt.grid(True)
    
plt.show()

from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# trained on linear kernel
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.20)


#clf = LinearSVC(random_state=42, tol=1e-3, max_iter=100000000)
#clf = NuSVC(nu = 0.9999999)
clf = SVC(kernel='poly', degree=3, gamma='auto')

clf.fit(X_train, y_train)
 
preds = clf.predict(X_test)
print(accuracy_score(y_test, preds))
                                                    



