import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a) Prikažite podatke za ucenje u x1−x2 ravnini matplotlib biblioteke pri cemu podatke obojite
# s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
# marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
# cmap kojima je moguce definirati boju svake klase.

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='Class 0')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='blue', label='Class 1')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='red', marker='x', label='Class 0 test')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='blue', marker='x', label='Class 1 test')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Train and test data')
plt.show()

# b)
# Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka
# za ucenje.

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# c)
# Pronadite u atributima izgradenog modela parametre modela. Prikažite granicu odluke
# naucenog modela u ravnini x1 −x2 zajedno s podacima za ucenje. Napomena: granica
# odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.

theta0 = log_reg.intercept_[0]
theta1, theta2 = log_reg.coef_[0]
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='Class 0')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='blue', label='Class 1')
plt.plot(X_train[:, 0], (-theta0 - theta1*X_train[:, 0]) / theta2, label='Decision boundary')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Train data and decision boundary')
plt.show()

# d)
# Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke
# regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izracunate tocnost,
# preciznost i odziv na skupu podataka za testiranje.

y_pred = log_reg.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ', conf_matrix)
disp= ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ' , recall)

# e)
# Prikazite skup za testiranje u ravnini x1−x2. Zelenom bojom oznacite dobro klasificirane
# primjere dok pogresno klasificirane primjere oznacite crnom bojom.

plt.scatter(X_test[y_test == y_pred, 0], X_test[y_test == y_pred, 1], c='green', label='Correctly classified')
plt.scatter(X_test[y_test != y_pred, 0], X_test[y_test != y_pred, 1], c='black', label='Misclassified')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Test data')
plt.show()
