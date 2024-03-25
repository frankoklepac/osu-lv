from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('LV4\data_C02_emission.csv')

# a)
X = dataframe[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
y = dataframe['CO2 Emissions (g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# b)
plt.figure()
plt.scatter(y_train, X_train['Fuel Consumption City (L/100km)'], color="blue",  s=10, alpha=0.5)
plt.scatter(y_test, X_test['Fuel Consumption City (L/100km)'], color="red",  s=10, alpha=0.5)
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Fuel Consumption City (L/100km)')
plt.show()

# c)
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train) 
scaled_X_train = pd.DataFrame(X_train_n, columns=X_train.columns) 
scaled_X_test = pd.DataFrame(X_train_n, columns=X_train.columns) 


X_train['Fuel Consumption City (L/100km)'].plot(kind='hist', bins=25)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('Frequency')
plt.show()
scaled_X_train['Fuel Consumption City (L/100km)'].plot(kind='hist', bins=25)
plt.xlabel('Train Fuel Consumption City (L/100km)')
plt.ylabel('Train Frequency')
plt.show()

# d)
print("==========TRAIN==========")
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)
print(linearModel.intercept_)

# e)
y_test_expect = linearModel.predict(sc.transform(X_test))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, X_test['Fuel Consumption City (L/100km)'], label='Real data', alpha=0.5, s=10)
plt.scatter(y_test_expect, X_test['Fuel Consumption City (L/100km)'], label='Predicted data', alpha=0.5, s=10)
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Fuel Consumption City (L/100km)')
plt.title('Real Data vs Predicted Data')
plt.legend()
plt.show()

#f)
print("==========TEST==========")
linearModel.fit(sc.transform(X_test), y_test)
print(linearModel.coef_)
print(linearModel.intercept_)

#g)
print("==========Original TEST==========")
linearModel.fit(sc.transform(X_test), y_test)
print(linearModel.coef_)
print(linearModel.intercept_)

print("==========1/2 TEST==========")
linearModel.fit(sc.transform(X_test[:int((len(X_test)-1)/2)]), y_test[:int((len(y_test)-1)/2)])
print(linearModel.coef_)
print(linearModel.intercept_)

print("==========1/4 TEST==========")
linearModel.fit(sc.transform(X_test[:int((len(X_test)-1)/4)]), y_test[:int((len(y_test)-1)/4)])
print(linearModel.coef_)
print(linearModel.intercept_)
