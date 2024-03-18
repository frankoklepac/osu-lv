import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('LV3\data_C02_emission.csv')

data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

grouped=data.groupby('Cylinders')
grouped.boxplot( column =['CO2 Emissions (g/km)'])
data.boxplot ( column =['CO2 Emissions (g/km)'], by='Cylinders')
plt.show ()