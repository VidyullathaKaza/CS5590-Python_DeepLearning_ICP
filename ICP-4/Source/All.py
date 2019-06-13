import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# load glass data set
glass = pd.read_csv('C:\\Users\\vidyu\\Desktop\\python code\\glass.csv')
x = glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = glass['Type']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = GaussianNB()
clf.fit(X_train, y_train)
print('Accuracy of Naive Bayes GaussianNB on training set: {:.3f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Naive Bayes GaussianNB on test set: {:.3f}'.format(clf.score(X_test, y_test)))


# Implement linear SVM method using scikit library
svm = LinearSVC(random_state=0, tol=1e-5)
svm.fit(X_train, y_train)
print('Accuracy of Linear SVM classifier on training set: {:.3f}'.format(svm.score(X_train, y_train)))
print('Accuracy of Linear SVM classifier on test set: {:.3f}'.format(svm.score(X_test, y_test)))


# Implement SVM method with RBF Kernel using scikit library
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print('Accuracy of SVM with RBF Kernel on training set: {:.3f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM with RBF Kernel on test set: {:.3f}'.format(svm.score(X_test, y_test)))