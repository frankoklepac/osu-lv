import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('LV3\data_C02_emission.csv')
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

# a) 
num_measurements = data.shape[0]
print('Broj mjerenja: ', num_measurements)

print('Tipovi podataka: ', data.dtypes)

missing_values = data.isnull().sum().sum()
print('Broj izostalih vrijednosti: ', missing_values)
if missing_values>0:
  data.dropna()

duplicated_rows = data.duplicated().sum()
print('Broj dupliciranih redaka: ', duplicated_rows)
if duplicated_rows>0:
  data.drop_duplicates()

# b)
data_sorted = data.sort_values('Fuel Consumption City (L/100km)')
lowest_consumption = data_sorted[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3)
highest_consumption = data_sorted[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3)

print('Najveca gradska potrosnja: ', highest_consumption)
print('Najmanja gradska potrosnja: ', lowest_consumption)

# c)
vehicles_filtered = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
vehicles_num = len(vehicles_filtered)
avg_co2_emission = vehicles_filtered['CO2 Emissions (g/km)'].mean()

print('Broj vozila sa velicinom motora između 2.5 i 3.5 L: ', vehicles_num)
print('Prosjecna CO2 emisija za ova vozila: ', avg_co2_emission)

# d)
audi_vehicles = data[data['Make'] == 'Audi']
audi_vehicles_num = len(audi_vehicles)
avg_co2_emission_audi_4_cyl = audi_vehicles[audi_vehicles['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean()

print('Auti marke Audi: ', audi_vehicles_num)
print('Prosjecna CO2 emisija za Audi automobile sa 4 cilindra: ', avg_co2_emission_audi_4_cyl)

# e)
cylinder_count = data['Cylinders'].value_counts()
avg_co2_emission_per_cylinder = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print('Broj vozila po broju cilindara: ', cylinder_count)
print('Prosjecna emisija CO2 plinova s obzirom na broj cilindara: ', avg_co2_emission_per_cylinder)

# f)


# g)

# h)

# i)
