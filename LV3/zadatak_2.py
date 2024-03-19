import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

# a)
plt.hist(data['CO2 Emissions (g/km)'], bins=50, color='skyblue')
plt.title('Histogram CO2 emisija')
plt.xlabel('CO2 emisiije (g/km)')
plt.ylabel('Frekvencija')
plt.show()

# b)
plt.scatter(data['Fuel Consumption City (L/100km)'], data['CO2 Emissions (g/km)'], c=data['Fuel Type'].cat.codes)
plt.title('City Fuel Consumption vs CO2 Emissions')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

# c)
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.title('Highway Fuel Consumption by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Fuel Consumption Hwy (L/100km)')
plt.show()

# d)
fuel_type_counts = data['Fuel Type'].value_counts()
fuel_type_counts.plot(kind='bar')
plt.title('Number of Vehicles by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Number of Vehicles')
plt.show()

# e)
avg_co2_emission_per_cylinder = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
avg_co2_emission_per_cylinder.plot(kind='bar')
plt.title('Average CO2 Emissions by Number of Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Average CO2 Emissions (g/km)')
plt.show()