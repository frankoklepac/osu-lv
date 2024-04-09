import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


# 1. Izradite algoritam KNN na skupu podataka za ucenje (uz K=5). Izracunajte tocnost
# klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. Usporedite
# dobivene rezultate s rezultatima logisticke regresije. Što primjecujete vezano uz dobivenu
# granicu odluke KNN modela?

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_n, y_train)

y_train_p_KNN = knn.predict(X_train_n)
y_test_p_KNN = knn.predict(X_test_n)

print("KNN (K=5) Accuracy, Training Set: {:0.3f}".format(accuracy_score(y_train, y_train_p_KNN)))
print("KNN (K=5) Accuracy, Testing Set: {:0.3f}".format(accuracy_score(y_test, y_test_p_KNN)))


# 2. Kako izgleda granica odluke kada je K = 1 i kada je K = 100?

for K in [1, 100]:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_n, y_train)
    y_train_pred = knn.predict(X_train_n)
    y_test_pred = knn.predict(X_test_n)
    print("KNN (K={}) Tocnost, Trening Set: {:0.3f}".format(K, accuracy_score(y_train, y_train_p_KNN)))
    print("KNN (K={}) Tocnost, Trening Set: {:0.3f}".format(K, accuracy_score(y_test, y_test_p_KNN)))
    plot_decision_regions(X_train_n, y_train, classifier=knn)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title("KNN (K={}) Accuracy: {:0.3f}".format(K, accuracy_score(y_train, y_train_p_KNN)))
    plt.tight_layout()
    plt.show()

# 6.5.2 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
# algoritma KNN za podatke iz Zadatka 1.

pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
param_grid = {'knn__n_neighbors': np.arange(1, 50)}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print("Najbolji parametri: ", grid.best_params_)
print("Tocnost: ", grid.best_score_)
print("Tocnost na testnom skupu: ", grid.score(X_test, y_test))

# Zadatak 6.5.3 Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
# te prikazite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
# ovih hiperparametara utjece na granicu odluke te pogresku na skupu podataka za testiranje?
# Mijenjajte tip kernela koji se koristi. Što primjecujete?

def apply_SVM(c, gamma):
    svm_model = svm.SVC(kernel='rbf', C=c, gamma=gamma)
    svm_model.fit(X_train_n, y_train)

    y_test_p_SVM = svm_model.predict(X_test_n)
    y_train_p_SVM = svm_model.predict(X_train_n)

    print("SVM: C: {}, gamma: {}".format(c, gamma))
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

    plot_decision_regions(X_train_n, y_train, classifier=svm_model)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
    plt.tight_layout()
    plt.show()

apply_SVM(1, 0.1)
apply_SVM(1, 1)
apply_SVM(10, 0.1)


# Zadatak 6.5.4 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
# algoritma SVM za problem iz Zadatka 1.

pipe = Pipeline([('scaler', StandardScaler()), ('svm', svm.SVC(kernel='rbf'))])
param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': [0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print("Najbolji parametri: ", grid.best_params_)
print("Tocnost: ", grid.best_score_)
print("Tocnost na testnom skupu: ", grid.score(X_test, y_test))

          
