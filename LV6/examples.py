from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

SVM_model = svm.SVC(kernel='rbf', gamma=1, C=0.1)
SVM_model.fit(X_train_n, y_train)

y_test_p_KNN = KNN_model.predict(X_test_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
