import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heatmap

#Read data into dataframe
DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF = DF.sort_values(by='Class')

DF['sisso1'] = abs((DF['Allred-Rochow EN']+DF['Atomic radius ratio'])-(DF['Allred-Rochow EN']*DF['Martynov-Batsanov Mean EN']))
DF['sisso2'] = abs((DF['Allred-Rochow Mean EN']*DF['Zungerradius sum ratio']) - (DF['Martynov Ionic Character']/DF['Atomic radius ratio']))

DF['binary1'] = (DF['Allred-Rochow EN']*DF['Ionic radius ratio']) * (DF['Martynov-Batsanov Mean EN'] - DF['Ionic radius ratio'])
DF['binary2'] = (DF['Allred-Rochow EN']*DF['Group number sum']) * (DF['Atomic radius sum'] - DF['Crystal radius sum'])

'''
less = DF.index[DF['binary1'] < 0.5].tolist()

for i in less:
	if DF.loc[i, 'Class'] == 1:
		print(DF.loc[i, 'compound'])

'''
plt.figure(figsize=(10, 10))
for i in [1, 2, 3, 4, 5, 6, 7]:
	index = DF.index[DF['Class'] == i].tolist()
	x = DF.loc[index, 'sisso1']
	y = DF.loc[index, 'sisso2']
	plt.scatter(x, y, s=10)

plt.legend(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'])
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

