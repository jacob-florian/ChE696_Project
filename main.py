import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heatmap

#Read data into dataframe
DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]

print(DF.tail(20))

#Plot Heatmap for some features
corr = DF.iloc[:, [5, 25, 30, 35]].corr()
plt.figure(figsize=(10, 10))
heatmap.corrplot(corr)

#Define features and target
X = DF.iloc[:, [5, 25]]
y = DF.iloc[:, 3]

#30/70 Test/Train stratified split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20, stratify=y)

#Standardize Features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
lr = KNeighborsClassifier(n_neighbors=10)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

############Visualize Model Performance############

#Function definitions
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

# Set-up grid for plotting.
X0, X1 = X_train_std[:, 0], X_train_std[:, 1]
X2, X3 = X_test_std[:, 0], X_test_std[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12,12))
# Plot the decision boundary. For that, we will assign a color to each
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# Plot training points
train = ax.scatter(X0, X1, c=y_train, cmap=plt.cm.Paired, s=30, edgecolor='k', label='training points')

# Plot test points
test = ax.scatter(X2, X3, c=y_test, cmap=plt.cm.Paired, s=30, edgecolor='w', label='test points')

ax.legend()
ax.set_ylabel('F2 [standardized]')
ax.set_xlabel('F1 [standardized]')
ax.set_xticks(())
ax.set_yticks(())


plt.savefig('2D_Model.png', dpi=300)
plt.show()

