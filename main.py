import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heatmap

#Read data into dataframe
DF = pd.read_csv('AB_data.csv', index_col=0)
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
columns = ['Number', 'Compound', 'Element1', 'Element2', 'Class', r'Pauling $χ_1-χ_2$', r'Martynov $χ_1-χ_2$', r'Gordy $χ_1-χ_2$', r'Mulliken $χ_1-χ_2$', r'Allred $χ_1-χ_2$', r'Pauling $χ_{avg}$', r'Martynov $χ_{avg}$', r'Gordy $χ_{avg}$', r'Mulliken $χ_{avg}$', r'Allred $χ_{avg}$', 'Pauling Ionic Character', 'Martynov Ionic Character', 'Gordy Ionic Character', 'Mulliken Ionic Character', 'Allred Ionic Character', 'Sum of Valence Electrons', 'Avg Number of Electrons', 'Atomic Number Sum', 'Atomic Number Difference', 'Avg Atomic Number','Atomic Weight Difference','Avg Atomic Weight','Atomic Weight Sum','D24','D25','D26','D27','D28','D29','D30','D31','D32','D33','D34','D35','D36','D37','D38','D39','D40','D41','D42','D43','D44','D45','D46','D47','D48','D49','D50','D51','D52','D53','D54','D55']

DF.columns = columns

corr = DF.iloc[:, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]].corr()

print(DF.tail())

#Plot Heatmap for some features
plt.figure(figsize=(10, 10))
heatmap.corrplot(corr)

plt.show()


