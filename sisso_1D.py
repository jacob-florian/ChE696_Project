import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heatmap

#Read data into dataframe
DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF = DF.sort_values(by='Class')

print(DF.tail(20))

DF['sisso1'] = np.exp(DF['Family number sum']) * (DF['Allred-Rochow EN'] / DF['Martynov Ionic Character'])
DF['sisso2'] = np.power(DF['Family number sum'], 3) * (DF['Allred-Rochow Mean EN'] / DF['Martynov Ionic Character'])

plt.figure(figsize=(10, 10))
for i in range(1, 8):
	index = DF.index[DF['Class'] == i].tolist()
	x = DF.loc[index, 'sisso1']
	y = DF.loc[index, 'sisso2']
	plt.scatter(x, y, s=5)

plt.legend(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'])
plt.xlabel('exp(Family Number Sum) * (Allred-Rochow EN/Martynov Ionic Character)')
plt.ylabel('(Family Number Sum)^3 * (Allred-Rochow EN/Martynov Ionic Character)')
plt.show()

