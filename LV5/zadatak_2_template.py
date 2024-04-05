import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# a)
# Pomocu stupcastog dijagrama prikazite koliko primjera postoji za svaku klasu (vrstu
# pingvina) u skupu podataka za ucenje i skupu podataka za testiranje. Koristite numpy
# funkciju unique.

unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, color='blue')
plt.xticks(unique, labels.values())
plt.title('Train data')
plt.show()

# b)
# Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka
# za ucenje.

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# c)
# Pronadite u atributima izgradenog modela parametre modela. Koja je razlika u odnosu na
# binarni klasifikacijski problem iz prvog zadatka?

theta0 = log_reg.intercept_
theta1, theta2 = log_reg.coef_[0]
print('theta0: ', theta0)
print('theta1: ', theta1)
print('theta2: ', theta2)

# d)
# Pozovite funkciju plot_decision_region pri cemu joj predajte podatke za ucenje i
# izgradeni model logisticke regresije. Kako komentirate dobivene rezultate?



# e)
# Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke
# regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izracunajte tocnost.
# Pomocu classification_report funkcije izracunajte vrijednost cetiri glavne metrike
# na skupu podataka za testiranje.

# f)
# Dodajte u model još ulaznih velicina. Što se dogada s rezultatima klasifikacije na skupu
# podataka za testiranje?